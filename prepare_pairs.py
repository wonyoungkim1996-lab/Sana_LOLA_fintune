"""Prepare complete English imageCaption/image pairs for SANA LoRA.

Input is a flat directory containing exactly one .jpg and one .json per
original basename. JSON has exactly one key, imageCaption. All pairs are kept,
including repeated images with different or identical captions. Books and
identical images are connected before assigning train/validation/test splits.
No model is loaded and no source file is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import unicodedata
import uuid

from image_io import inspect_image
from prepare_data import PRIORITY, UnionFind, identities

HERE = Path(__file__).resolve().parent
NAME = re.compile(r"^(?P<category>\d{2})_(?P<text>\d{2}T)_(?P<scene>\d{2}S)_"
                  r"(?P<isbn>\d{13}|\d{9}[\dXx])_(?P<record>\d+)$")
SPLIT_FILES = ("train.jsonl", "validation.jsonl", "test.jsonl", "report.json")


class PreparationError(ValueError):
    """Input validation failed; report.json describes every detected issue."""

    def __init__(self, report):
        self.report = report
        super().__init__("Pair preparation failed: " + json.dumps(report["issue_counts"], sort_keys=True))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value")))


def read_jsonl(path):
    raw = Path(path).read_bytes()
    rows = []
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if line.strip():
            row = parse_json(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(row)
    return rows, sha(raw)


def canonical_isbn(value):
    """Validate ISBN-10/13 checksum; return the equivalent ISBN-13."""
    value = re.sub(r"[\s-]", "", str(value)).upper()
    if re.fullmatch(r"\d{9}[\dX]", value):
        total = sum((10 if char == "X" else int(char)) * (10 - index)
                    for index, char in enumerate(value))
        if total % 11:
            raise ValueError("Invalid ISBN-10 checksum")
        prefix = "978" + value[:9]
        check = (-sum(int(char) * (1 if index % 2 == 0 else 3)
                      for index, char in enumerate(prefix))) % 10
        return prefix + str(check)
    if re.fullmatch(r"97[89]\d{10}", value):
        if sum(int(char) * (1 if index % 2 == 0 else 3)
               for index, char in enumerate(value)) % 10:
            raise ValueError("Invalid ISBN-13 checksum")
        return value
    raise ValueError("Expected a valid ISBN-10 or 978/979 ISBN-13")


def filename_identity(stem):
    match = NAME.fullmatch(stem)
    if match is None:
        raise ValueError("Unsupported basename; expected NN_NNT_NNS_ISBN_RECORD")
    result = match.groupdict()
    result["filename_isbn"] = result["isbn"].upper()
    result["isbn"] = canonical_isbn(result["isbn"])
    return result


def english_issues(caption):
    if not isinstance(caption, str) or not caption.strip():
        return ["caption_not_nonempty_string"]
    issues = []
    if not any(char.isalpha() and "LATIN" in unicodedata.name(char, "") for char in caption):
        issues.append("caption_has_no_latin_letters")
    if any(char.isalpha() and "LATIN" not in unicodedata.name(char, "") for char in caption):
        issues.append("caption_contains_non_latin_script")
    if re.search(r"</?think>|```|<\||(?:^|\n)\s*(?:translation|english translation|note|explanation)\s*:", caption, re.I):
        issues.append("caption_contains_translation_metadata")
    if caption.lstrip().startswith(("{", "[")):
        issues.append("caption_contains_nested_structure")
    return issues


def source_stem(path):
    # A metadata manifest produced on Windows can be used on Linux without
    # treating the entire Windows absolute path as one POSIX filename.
    value = str(path)
    return PureWindowsPath(value).stem if "\\" in value else Path(value).stem


def linked_identities(row):
    keys = identities(row)
    if row.get("isbn"):
        try:
            keys.append("isbn:" + canonical_isbn(row["isbn"]))
        except ValueError:
            pass  # Existing anchors can still link by title/context/group.
    return list(dict.fromkeys(keys))


def write_outputs(output_dir, rows_by_split, report):
    """Publish files only after validation. report.json is published last."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {}
    if rows_by_split is not None:
        for split, rows in rows_by_split.items():
            payloads[split + ".jsonl"] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    payloads["report.json"] = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    for name, text in payloads.items():
        target = output_dir / name
        temporary = target.with_name("." + name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(pair_dir, output_dir, *, expected_count=40001, source_manifest=None,
            existing_split_dir=None, workers=2, max_image_pixels=300_000_000,
            seed=20260915, validation_ratio=.1, test_ratio=.1, overwrite=False):
    pair_dir, output_dir = Path(pair_dir).resolve(), Path(output_dir).resolve()
    source_manifest = Path(source_manifest).resolve() if source_manifest else None
    existing_split_dir = Path(existing_split_dir).resolve() if existing_split_dir else None
    if expected_count < 1 or not 1 <= workers <= 32 or max_image_pixels < 1:
        raise ValueError("expected-count/max-image-pixels must be positive; workers must be 1..32")
    if min(validation_ratio, test_ratio) < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("Split ratios must be nonnegative and sum to less than one")
    if output_dir.is_relative_to(pair_dir):
        raise ValueError("Output directory must be outside the original pair directory")
    protected = {source_manifest} if source_manifest else set()
    if existing_split_dir:
        protected.update(existing_split_dir / (split + ".jsonl") for split in PRIORITY)
    outputs = {output_dir / name for name in SPLIT_FILES}
    if outputs & protected:
        raise ValueError("Output paths would overwrite a source manifest or existing split")
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Prepared outputs exist; choose a new output directory or explicit --overwrite")

    report = {
        "format_version": 1, "complete": False, "split_files_written": False,
        "pair_dir": str(pair_dir), "output_dir": str(output_dir), "expected_pairs": expected_count,
        "source_manifest": str(source_manifest) if source_manifest else None,
        "existing_split_dir": str(existing_split_dir) if existing_split_dir else None,
        "seed": seed, "workers": workers, "max_image_pixels": max_image_pixels,
        "counts": {}, "issues": [], "issue_counts": {}, "provenance": {},
        "policies": {
            "caption": "Exact English imageCaption from the paired JSON; no Korean fallback or metadata augmentation",
            "input": "Flat directory, exact case-sensitive .jpg/.json names, no missing/extra/duplicate casefolded basenames",
            "retention": "Keep every valid filename pair, including all alternative and repeated captions for identical images",
            "group_links": "Checksum-validated ISBN-10/13 book identity plus identical encoded bytes or oriented RGB pixels; optional title/context/existing Qwen group links",
            "isbn": "ISBN-10 is converted to its equivalent ISBN-13 for book grouping",
            "existing_split_precedence": "test > validation > train",
            "new_group_split": "SHA256(seed:component_id) threshold; ratios apply to groups approximately, not exact image counts",
            "ratios": {"train": 1-validation_ratio-test_ratio, "validation": validation_ratio, "test": test_ratio},
        },
        "warnings": [
            "English script and file integrity checks do not prove faithful translation or caption-image semantic agreement.",
            "Exact hashes do not identify all resized, recompressed or visually similar images.",
            "Repeated images and alternative captions are retained as requested and therefore receive repeated training weight.",
            "Inherited split conflicts are promoted conservatively; this cannot undo prior model exposure to a held-out book.",
        ],
        "semantic_translation_quality_verified": False, "caption_image_semantic_accuracy_verified": False,
    }
    def issue(reason, **details):
        report["issues"].append({"reason": reason, **details})
    def fail_if_needed():
        if report["issues"]:
            report["issue_counts"] = dict(Counter(row["reason"] for row in report["issues"]))
            write_outputs(output_dir, None, report)
            raise PreparationError(report)

    if not pair_dir.is_dir():
        issue("missing_pair_directory")
        fail_if_needed()
    entries = sorted(pair_dir.iterdir(), key=lambda path: (path.name.casefold(), path.name))
    by_folded_name = defaultdict(list)
    images, labels = {}, {}
    for path in entries:
        by_folded_name[path.name.casefold()].append(path.name)
        if not path.is_file():
            issue("non_file_entry", name=path.name)
        elif path.suffix == ".jpg":
            images[path.stem] = path
        elif path.suffix == ".json":
            labels[path.stem] = path
        else:
            issue("unexpected_file", name=path.name)
    for name, values in by_folded_name.items():
        if len(values) > 1:
            issue("duplicate_casefolded_basename", names=values)
    for stem in sorted(set(images) - set(labels)):
        issue("missing_json", stem=stem)
    for stem in sorted(set(labels) - set(images)):
        issue("missing_image", stem=stem)
    report["counts"].update(input_entries=len(entries), image_files=len(images), json_files=len(labels),
                             exact_filename_pairs=len(set(images) & set(labels)), kept_rows=0)
    if len(images) != expected_count or len(labels) != expected_count or len(entries) != expected_count * 2:
        issue("unexpected_pair_count", expected=expected_count, images=len(images), jsons=len(labels), entries=len(entries))
    report["provenance"]["input_filenames_sha256"] = sha("\n".join(path.name for path in entries).encode("utf-8"))

    metadata = {}
    if source_manifest:
        try:
            source_rows, source_hash = read_jsonl(source_manifest)
            report["provenance"]["source_manifest_sha256"] = source_hash
            folded = set()
            for number, row in enumerate(source_rows, 1):
                if not isinstance(row.get("image"), str) or not row["image"]:
                    raise ValueError(f"Manifest row {number} has no image path")
                stem = source_stem(row["image"])
                if stem.casefold() in folded:
                    raise ValueError(f"Duplicate manifest image basename: {stem}")
                folded.add(stem.casefold())
                metadata[stem] = row
            report["counts"]["source_manifest_rows"] = len(source_rows)
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            issue("invalid_source_manifest", detail=str(exc))
    candidates = []
    for stem in sorted(set(images) & set(labels)):
        try:
            name_info = filename_identity(stem)
        except ValueError as exc:
            issue("invalid_filename_or_isbn", stem=stem, detail=str(exc))
            continue
        try:
            raw_label = labels[stem].read_bytes()
            parsed = parse_json(raw_label.decode("utf-8-sig"))
            if not isinstance(parsed, dict) or list(parsed) != ["imageCaption"]:
                raise ValueError("JSON must contain exactly one key: imageCaption")
            caption = parsed["imageCaption"]
        except (OSError, UnicodeError, ValueError, TypeError) as exc:
            issue("invalid_caption_json", stem=stem, detail=str(exc))
            continue
        invalid = english_issues(caption)
        if invalid:
            for reason in invalid:
                issue(reason, stem=stem)
            continue
        source = metadata.get(stem, {})
        if source_manifest and stem not in metadata:
            issue("source_manifest_missing_exact_stem", stem=stem)
            continue
        if source.get("isbn"):
            try:
                if canonical_isbn(source["isbn"]) != name_info["isbn"]:
                    raise ValueError("Manifest ISBN does not match filename ISBN")
            except ValueError as exc:
                issue("source_manifest_isbn_mismatch", stem=stem, detail=str(exc))
                continue
        candidates.append({"stem": stem, "image": images[stem], "label_json": labels[stem],
                           "caption": caption, "caption_sha256": sha(caption.encode("utf-8")),
                           "label_json_sha256": sha(raw_label), "source": source, **name_info})
    report["counts"]["valid_caption_rows"] = len(candidates)
    fail_if_needed()  # No expensive image decoding if pairing/caption input is invalid.

    uf = UnionFind()
    existing_counts = Counter()
    if existing_split_dir:
        for split in PRIORITY:
            path = existing_split_dir / (split + ".jsonl")
            try:
                anchor_rows, anchor_hash = read_jsonl(path)
                report["provenance"]["existing_" + split + "_sha256"] = anchor_hash
                for anchor in anchor_rows:
                    keys = linked_identities(anchor)
                    if keys:
                        uf.union(keys, split)
                existing_counts[split] = len(anchor_rows)
            except (OSError, UnicodeError, ValueError, TypeError) as exc:
                issue("invalid_existing_split", split=split, detail=str(exc))
    fail_if_needed()
    for row in candidates:
        row["key"] = "pair:" + row["stem"]
        merged_metadata = {**row["source"], "isbn": row["isbn"]}
        uf.union([row["key"], "isbn:" + row["filename_isbn"]] + linked_identities(merged_metadata))

    def inspect(row):
        return inspect_image(row["image"], max_pixels=max_image_pixels)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row, info in zip(candidates, executor.map(inspect, candidates)):
            if "error" in info:
                issue("image_decode_or_integrity_error", stem=row["stem"], detail=info["error"])
                continue
            row["info"] = info
            uf.union([row["key"], "image-bytes:" + info["image_sha256"], "image-pixels:" + info["pixel_sha256"]])
    report["counts"]["decoded_rows"] = sum("info" in row for row in candidates)
    fail_if_needed()

    components = defaultdict(list)
    for key in list(uf.parent):
        if not key.startswith("pair:"):
            components[uf.find(key)].append(key)
    group_ids = {root: sha("\n".join(sorted(keys)).encode("utf-8"))[:24] for root, keys in components.items()}
    split_rows = {split: [] for split in PRIORITY}
    inherited, mixed_existing = Counter(), {}
    byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
    for row in candidates:
        info = row["info"]
        root = uf.find(row["key"])
        group = group_ids[root]
        tags = uf.tags[root]
        if tags:
            split = max(tags, key=PRIORITY.get)
            inherited[split] += 1
            if len(tags) > 1:
                mixed_existing[group] = {"existing_splits": sorted(tags), "selected": split}
        else:
            draw = int(sha(f"{seed}:{group}".encode("utf-8"))[:16], 16) / 2**64
            split = "test" if draw < test_ratio else "validation" if draw < test_ratio + validation_ratio else "train"
        record = {
            "id": sha((row["stem"] + "\0" + row["caption_sha256"] + "\0" + info["image_sha256"]).encode("utf-8"))[:24],
            "image": str(row["image"]), "caption": row["caption"], "group": group, "split": split,
            "isbn": row["isbn"], "filename_isbn": row["filename_isbn"], "stem": row["stem"],
            "title": str(row["source"].get("title") or ""),
            "source_text": str(row["source"].get("source_text", row["source"].get("context", "")) or ""),
            "label_json": str(row["label_json"]), "label_json_sha256": row["label_json_sha256"],
            "caption_sha256": row["caption_sha256"],
            **info,
        }
        split_rows[split].append(record)
        byte_groups[info["image_sha256"]].append(record)
        pixel_groups[info["pixel_sha256"]].append(record)
    all_records = [row for rows in split_rows.values() for row in rows]
    if len(all_records) != expected_count or len({row["id"] for row in all_records}) != expected_count:
        raise RuntimeError("Internal row retention or ID uniqueness check failed")
    for key in ("group", "isbn", "image_sha256", "pixel_sha256"):
        seen = {}
        for row in all_records:
            if row[key] in seen and seen[row[key]] != row["split"]:
                raise RuntimeError(f"Internal split leakage check failed for {key}")
            seen[row[key]] = row["split"]
    groups = {split: {row["group"] for row in rows} for split, rows in split_rows.items()}
    report["counts"].update(
        kept_rows=len(all_records), excluded_rows=0,
        split_rows={split: len(rows) for split, rows in split_rows.items()},
        split_groups={split: len(values) for split, values in groups.items()},
        canonical_book_isbns=len({row["isbn"] for row in all_records}),
        identical_byte_groups=sum(len(rows)>1 for rows in byte_groups.values()),
        identical_pixel_groups=sum(len(rows)>1 for rows in pixel_groups.values()),
        identical_pixel_extra_rows_retained=sum(len(rows)-1 for rows in pixel_groups.values()),
        identical_pixel_groups_with_alternative_captions=sum(len({row["caption"] for row in rows})>1 for rows in pixel_groups.values()),
        identical_pixel_same_caption_extra_rows_retained=sum(len(rows)-len({row["caption"] for row in rows}) for rows in pixel_groups.values()),
        exif_transposed_images=sum(bool(row.get("exif_transposed")) for row in all_records),
        alpha_composited_images=sum(bool(row.get("alpha_composited")) for row in all_records),
        large_image_override_images=sum(bool(row.get("large_image_override")) for row in all_records),
    )
    report["existing_rows_indexed"] = dict(existing_counts)
    report["inherited_split_rows"] = dict(inherited)
    report["existing_split_conflicts"] = mixed_existing
    report["split_leakage_checks"] = {key: "passed" for key in ("group", "isbn", "image_sha256", "pixel_sha256")}
    provenance_rows = [{key: row[key] for key in ("stem", "image_sha256", "pixel_sha256", "label_json_sha256", "caption_sha256")}
                       for row in sorted(all_records, key=lambda value: value["stem"])]
    report["provenance"]["ordered_pair_content_sha256"] = sha(json.dumps(provenance_rows, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    report["provenance"]["pixel_hash_policy"] = "Shared image_io: EXIF-oriented RGB, alpha composited on white; original source bytes unchanged"
    if not existing_split_dir:
        report["warnings"].append("Independent book split: coordinate with prior LLM evaluation books before comparing an end-to-end system.")
    if any(not rows for rows in split_rows.values()):
        report["warnings"].append("One or more splits are empty; book/image isolation takes precedence over requested approximate ratios.")
    report.update(complete=True, split_files_written=True)
    write_outputs(output_dir, split_rows, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, required=True, help="Flat directory of originalstem.jpg + originalstem.json English pairs")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data", help="Prepared JSONL directory outside --pair-dir")
    parser.add_argument("--expected-count", type=int, default=40001, help="Exact required pair count; smaller values are for explicit smaller datasets/tests")
    parser.add_argument("--source-manifest", type=Path, help="Optional metadata JSONL joined by exact image stem; captions are never read from it")
    parser.add_argument("--existing-split-dir", type=Path, help="Optional directory containing all three existing Qwen split JSONL files")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-image-pixels", type=int, default=300_000_000, help="Explicit image decoding safety cap, shared with image_io")
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--validation-ratio", type=float, default=.1)
    parser.add_argument("--test-ratio", type=float, default=.1)
    parser.add_argument("--overwrite", action="store_true", help="Replace generated split/report files only; originals are never edited")
    args = parser.parse_args()
    try:
        report = prepare(**vars(args))
    except PreparationError as exc:
        print(json.dumps({"complete": False, "issue_counts": exc.report["issue_counts"],
                          "report": str(args.output_dir.resolve() / "report.json")}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"complete": report["complete"], "counts": report["counts"],
                      "report": str(args.output_dir.resolve() / "report.json")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
