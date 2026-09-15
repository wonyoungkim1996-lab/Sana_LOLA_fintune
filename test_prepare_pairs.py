"""Small synthetic CPU tests; never scan or modify the real corpus."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_pairs


BOOK_A = "9780306406157"
BOOK_A10 = "0306406152"
BOOK_B = "9781861972712"


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def output_rows(directory):
    rows = []
    for split in ("train", "validation", "test"):
        for line in (directory / (split + ".jsonl")).read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line))
    return rows


class PreparePairsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pairs = self.root / "pairs"
        self.pairs.mkdir()
        self.output = self.root / "prepared"

    def pair(self, isbn=BOOK_A, record=1, caption="A child opens a red door.", color="red", payload=None):
        stem = f"03_01T_01S_{isbn}_{record}"
        image, label = self.pairs / (stem + ".jpg"), self.pairs / (stem + ".json")
        if payload is None:
            Image.new("RGB", (24, 16), color=color).save(image, format="JPEG")
        else:
            image.write_bytes(payload)
        label.write_text(json.dumps({"imageCaption": caption}, ensure_ascii=False), encoding="utf-8")
        return image, label

    def prepare(self, count, **kwargs):
        options = {"expected_count": count, "workers": 1, "validation_ratio": 0., "test_ratio": 0.}
        options.update(kwargs)
        return prepare_pairs.prepare(self.pairs, self.output, **options)

    def assert_failed(self, count, reason):
        with self.assertRaises(prepare_pairs.PreparationError) as caught:
            self.prepare(count)
        report = caught.exception.report
        self.assertFalse(report["complete"])
        self.assertFalse(report["split_files_written"])
        self.assertIn(reason, report["issue_counts"])
        self.assertTrue((self.output / "report.json").is_file())
        self.assertFalse((self.output / "train.jsonl").exists())

    def test_every_pair_and_alternative_caption_is_retained_with_image_grouping(self):
        first, _ = self.pair(record=1, caption="A red square.")
        self.pair(record=2, caption="The same red square.", payload=first.read_bytes())
        # Different bytes, identical decoded pixels, different book, repeated caption.
        self.pair(isbn=BOOK_B, record=3, caption="A red square.", payload=first.read_bytes() + b"\n")
        self.pair(record=4, caption="A blue square.", color="blue")
        original = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.pairs.iterdir()}
        report = self.prepare(4)
        rows = output_rows(self.output)
        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["kept_rows"], 4)
        self.assertEqual(len({row["id"] for row in rows}), 4)
        self.assertEqual(len({row["group"] for row in rows}), 1)
        self.assertEqual(report["counts"]["identical_pixel_extra_rows_retained"], 2)
        self.assertEqual(report["counts"]["identical_pixel_groups_with_alternative_captions"], 1)
        self.assertEqual(report["counts"]["identical_pixel_same_caption_extra_rows_retained"], 1)
        self.assertEqual(original, {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.pairs.iterdir()})
        for row in rows:
            self.assertEqual(row["caption"], json.loads(Path(row["label_json"]).read_text(encoding="utf-8"))["imageCaption"])
            for key in ("image_sha256", "pixel_sha256", "caption_sha256", "label_json_sha256"):
                self.assertEqual(len(row[key]), 64)
        self.assertFalse(report["semantic_translation_quality_verified"])

    def test_isbn10_and_isbn13_same_book_have_one_group_and_stable_split(self):
        self.assertEqual(prepare_pairs.canonical_isbn(BOOK_A10), BOOK_A)
        self.assertEqual(prepare_pairs.canonical_isbn("080442957X"), "9780804429573")
        self.pair(isbn=BOOK_A10, record=1, color="red")
        self.pair(isbn=BOOK_A, record=2, color="blue")
        self.pair(isbn=BOOK_B, record=3, color="green")
        report = self.prepare(3, seed=19, validation_ratio=.3, test_ratio=.3)
        rows = output_rows(self.output)
        equivalent = [row for row in rows if row["isbn"] == BOOK_A]
        self.assertEqual(len(equivalent), 2)
        self.assertEqual(len({row["group"] for row in equivalent}), 1)
        self.assertEqual(len({row["split"] for row in equivalent}), 1)
        other_output = self.root / "prepared_again"
        prepare_pairs.prepare(self.pairs, other_output, expected_count=3, workers=2,
                              seed=19, validation_ratio=.3, test_ratio=.3)
        self.assertEqual(sorted(rows, key=lambda row: row["id"]), sorted(output_rows(other_output), key=lambda row: row["id"]))
        self.assertEqual(report["counts"]["canonical_book_isbns"], 2)

    def test_source_metadata_context_and_existing_split_priority_do_not_replace_caption(self):
        image_a, _ = self.pair(record=1, color="red", caption="  A red door.  ")
        image_b, _ = self.pair(isbn=BOOK_B, record=2, color="blue", caption="A blue door.")
        manifest = self.root / "metadata.jsonl"
        metadata = []
        for image, isbn in ((image_a, BOOK_A10), (image_b, BOOK_B)):
            metadata.append({"image": "Z:\\old_dataset\\" + image.name, "isbn": isbn,
                             "title": "원본 도서", "source_text": "Shared source context.",
                             "caption": "이 한국어 캡션은 사용하면 안 됩니다."})
        jsonl(manifest, metadata)
        anchors = self.root / "existing"
        anchors.mkdir()
        jsonl(anchors / "train.jsonl", [{"isbn": BOOK_A10, "group": "old_train"}])
        jsonl(anchors / "validation.jsonl", [{"isbn": BOOK_B, "group": "old_validation"}])
        jsonl(anchors / "test.jsonl", [{"context": "Shared source context.", "group": "old_test"}])
        report = self.prepare(2, source_manifest=manifest, existing_split_dir=anchors)
        rows = output_rows(self.output)
        self.assertEqual({row["split"] for row in rows}, {"test"})
        self.assertEqual(len({row["group"] for row in rows}), 1)
        self.assertEqual({row["caption"] for row in rows}, {"  A red door.  ", "A blue door."})
        self.assertEqual(report["inherited_split_rows"], {"test": 2})
        self.assertEqual(len(report["existing_split_conflicts"]), 1)

    def test_missing_pair_or_unexpected_count_is_a_hard_failure(self):
        self.pair()
        self.assert_failed(2, "unexpected_pair_count")

    def test_orphan_or_case_mismatched_json_is_a_hard_failure(self):
        image, label = self.pair()
        renamed = label.with_name(label.name.replace("01T", "01t"))
        label.rename(renamed)
        self.assert_failed(1, "missing_json")
        self.assertIn("missing_image", json.loads((self.output / "report.json").read_text(encoding="utf-8"))["issue_counts"])

    def test_broken_image_is_rejected_without_partial_split_exports(self):
        self.pair(payload=b"not an image")
        self.assert_failed(1, "image_decode_or_integrity_error")

    def test_casefolded_basename_collisions_and_extra_files_are_rejected(self):
        image, label = self.pair()
        # NTFS normally prevents storing these two names simultaneously. A
        # synthetic listing also exercises the collision check on Windows;
        # a case-sensitive source filesystem can contain this actual layout.
        alias = image.with_suffix(".JPG")
        with mock.patch.object(Path, "iterdir", return_value=iter([image, label, alias])):
            self.assert_failed(1, "duplicate_casefolded_basename")
        self.output = self.root / "extra_file_output"
        (self.pairs / "notes.txt").write_text("Not a training pair.", encoding="utf-8")
        self.assert_failed(1, "unexpected_file")

    def test_unsupported_name_and_invalid_isbn_are_not_split_as_single_images(self):
        image, label = self.pair(isbn="9780306406158")
        self.assert_failed(1, "invalid_filename_or_isbn")
        image.rename(self.pairs / "unknown_scene.jpg")
        label.rename(self.pairs / "unknown_scene.json")
        self.output = self.root / "unsupported_name_output"
        self.assert_failed(1, "invalid_filename_or_isbn")

    def test_english_only_exact_json_schema_and_duplicate_keys_are_required(self):
        _, label = self.pair()
        cases = [
            ('{"imageCaption":"A door.","metadata":"extra"}', "invalid_caption_json"),
            ('{"imageCaption":"A door.","imageCaption":"Another door."}', "invalid_caption_json"),
            ('{"imageCaption":"A 문 opens."}', "caption_contains_non_latin_script"),
            ('{"imageCaption":"A 门 opens."}', "caption_contains_non_latin_script"),
            ('{"imageCaption":""}', "caption_not_nonempty_string"),
        ]
        for index, (content, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                label.write_text(content, encoding="utf-8")
                self.output = self.root / f"failed_{index}"
                self.assert_failed(1, reason)

    def test_metadata_cannot_silently_attach_another_book_or_fallback_caption(self):
        image, _ = self.pair()
        manifest = self.root / "metadata.jsonl"
        jsonl(manifest, [{"image": str(image), "isbn": BOOK_B, "caption": "A fallback caption."}])
        with self.assertRaises(prepare_pairs.PreparationError) as caught:
            self.prepare(1, source_manifest=manifest)
        self.assertIn("source_manifest_isbn_mismatch", caught.exception.report["issue_counts"])

    def test_original_and_existing_outputs_are_protected(self):
        image, label = self.pair()
        with self.assertRaises(ValueError):
            prepare_pairs.prepare(self.pairs, self.pairs / "generated", expected_count=1)
        self.prepare(1)
        original = (self.output / "train.jsonl").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare(1)
        self.assertEqual((self.output / "train.jsonl").read_bytes(), original)
        self.assertTrue(image.is_file())
        self.assertEqual(list(json.loads(label.read_text(encoding="utf-8"))), ["imageCaption"])


if __name__ == "__main__":
    unittest.main()
