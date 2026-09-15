"""Prepare caption-conditioned image pairs without modifying source data.

Uses only the standard library and Pillow. Existing Qwen split assignments are
inherited conservatively: test > validation > train. --limit is a partial
development preview and must not be used as a full-corpus leakage audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata
import warnings

from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PRIORITY = {"train": 0, "validation": 1, "test": 2}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def identities(row):
    values = []
    isbn = re.sub(r"[^0-9X]", "", str(row.get("isbn") or "").upper())
    title = "".join(c.lower() for c in normalized(row.get("title")) if c.isalnum())
    context = "".join(normalized(row.get("context", row.get("source_text", ""))).split())
    if isbn:
        values.append("isbn:" + isbn)
    if title:
        values.append("title:" + title)
    if context:
        values.append("context:" + digest(context.encode("utf-8")))
    if row.get("group"):
        values.append("existing-group:" + str(row["group"]))
    return values


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.tags = {}

    def find(self, key):
        if key not in self.parent:
            self.parent[key] = key
            self.tags[key] = set()
        trail = []
        while key != self.parent[key]:
            trail.append(key)
            key = self.parent[key]
        for item in trail:
            self.parent[item] = key
        return key

    def union(self, keys, split=None):
        keys = list(keys)
        if not keys:
            raise ValueError("Cannot union an empty identity set")
        root = self.find(keys[0])
        for key in keys[1:]:
            other = self.find(key)
            if root != other:
                self.parent[other] = root
                self.tags[root].update(self.tags.pop(other))
        if split:
            self.tags[root].add(split)
        return root


def jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected JSON object")
                yield row


def image_path(value, image_root):
    path = Path(value)
    return (path if path.is_absolute() else image_root / path).resolve()


def inspect_image(path):
    """Verify encoded image and decode oriented pixels, without saving edits."""
    try:
        data = path.read_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as original:
                original.verify()
            with Image.open(BytesIO(data)) as original:
                orientation = original.getexif().get(274, 1)
                oriented = ImageOps.exif_transpose(original)
                oriented.load()
                rgb = oriented.convert("RGB")
                width, height = rgb.size
                pixels_hash = hashlib.sha256()
                pixels_hash.update(f"RGB:{width}:{height}:".encode("ascii"))
                pixels_hash.update(rgb.tobytes())
        if width <= 0 or height <= 0:
            raise ValueError("empty image dimensions")
        return {"image_sha256": digest(data), "pixel_sha256": pixels_hash.hexdigest(),
                "width": width, "height": height, "exif_transposed": orientation not in (None, 1)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def prepare(manifest, output_dir, existing_split_dir, image_root=None, *, limit=None,
            workers=4, overwrite=False, caption_overrides=None, seed=20260914,
            validation_ratio=.1, test_ratio=.1):
    manifest, output_dir = map(Path, (manifest, output_dir))
    existing_split_dir = Path(existing_split_dir) if existing_split_dir is not None else None
    image_root = Path(image_root or WORKSPACE).resolve()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if workers < 1 or workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if min(validation_ratio, test_ratio) < 0 or validation_ratio + test_ratio >= 1:
        raise ValueError("validation/test ratios must be nonnegative with sum < 1")
    outputs = [output_dir / name for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "report.json")]
    protected = {manifest.resolve()}
    if existing_split_dir is not None:
        protected.update((existing_split_dir / (split + ".jsonl")).resolve() for split in PRIORITY)
    if caption_overrides:
        protected.add(Path(caption_overrides).resolve())
    if any(path.resolve() in protected for path in outputs):
        raise ValueError("Output paths would overwrite source manifest or existing split data")
    if not overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("Prepared outputs exist; use a new output directory or explicit --overwrite")
    uf = UnionFind()
    existing_counts = Counter()
    if existing_split_dir is not None:
        for split in PRIORITY:
            path = existing_split_dir / (split + ".jsonl")
            if not path.is_file():
                raise FileNotFoundError(f"Required existing split file missing: {path}")
            for index, row in enumerate(jsonl(path)):
                keys = identities(row)
                if keys:
                    uf.union(keys, split)
                existing_counts[split] += 1
    overrides = {}
    if caption_overrides:
        for row in jsonl(caption_overrides):
            if not isinstance(row.get("image"), str) or not normalized(row.get("caption")):
                raise ValueError("Caption override rows require image and nonempty caption")
            key = os.path.normcase(str(image_path(row["image"], image_root)))
            if key in overrides and overrides[key] != row["caption"]:
                raise ValueError(f"Conflicting caption overrides: {key}")
            overrides[key] = row["caption"]
    selected = []
    for index, row in enumerate(jsonl(manifest)):
        if limit is not None and index >= limit:
            break
        selected.append(row)
    exclusions, candidates = [], []
    override_applied = 0
    for index, row in enumerate(selected):
        if not isinstance(row.get("image"), str) or not row["image"].strip():
            exclusions.append({"row": index + 1, "reason": "missing_image_path"})
            continue
        path = image_path(row["image"], image_root)
        key = "manifest-row:" + str(index)
        uf.union([key] + identities(row))
        override_key = os.path.normcase(str(path))
        caption = overrides.get(override_key, row.get("caption"))
        if override_key in overrides:
            override_applied += 1
        valid_caption = isinstance(caption, str) and bool(normalized(caption))
        if not valid_caption:
            exclusions.append({"row": index + 1, "reason": "empty_caption", "image": str(path)})
        candidates.append({"source": row, "index": index, "key": key, "path": path, "caption": caption,
                           "valid_caption": valid_caption})
    unique_paths = list(dict.fromkeys(row["path"] for row in candidates))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected = dict(zip(unique_paths, executor.map(inspect_image, unique_paths)))
    valid = []
    for row in candidates:
        info = inspected[row["path"]]
        if "error" in info:
            exclusions.append({"row": row["index"] + 1, "reason": "image_decode_or_read_error",
                               "image": str(row["path"]), "detail": info["error"]})
            continue
        row["info"] = info
        uf.union([row["key"], "image-bytes:" + info["image_sha256"], "image-pixels:" + info["pixel_sha256"]])
        if not row["valid_caption"]:
            continue  # Even excluded captions keep image/book links for split isolation.
        valid.append(row)
    by_pixels = defaultdict(list)
    for row in valid:
        by_pixels[row["info"]["pixel_sha256"]].append(row)
    kept = []
    conflicts = []
    duplicate_count = 0
    for image_hash, rows in by_pixels.items():
        captions = {normalized(row["caption"]) for row in rows}
        if len(captions) > 1:
            conflicts.append({"pixel_sha256": image_hash, "rows": [r["index"] + 1 for r in rows],
                              "distinct_captions": len(captions)})
            for row in rows:
                exclusions.append({"row": row["index"] + 1, "reason": "same_image_conflicting_captions", "image": str(row["path"])})
            continue
        rows.sort(key=lambda row: row["index"])
        kept.append(rows[0])
        for row in rows[1:]:
            duplicate_count += 1
            exclusions.append({"row": row["index"] + 1, "reason": "duplicate_image_same_caption", "image": str(row["path"])})
    # Derive deterministic group IDs after every book/context/image union.
    components = defaultdict(list)
    for key in uf.parent:
        root = uf.find(key)
        if not key.startswith("manifest-row:"):
            components[root].append(key)
    group_ids = {root: digest("\n".join(sorted(keys)).encode("utf-8"))[:20] for root, keys in components.items()}
    split_rows = {split: [] for split in PRIORITY}
    inherited = Counter()
    mixed_existing = {}
    for row in sorted(kept, key=lambda item: item["index"]):
        root = uf.find(row["key"])
        group = group_ids[root]
        tags = uf.tags[root]
        if tags:
            split = max(tags, key=PRIORITY.get)
            inherited[split] += 1
            if len(tags) > 1:
                mixed_existing[group] = {"existing_splits": sorted(tags), "selected": split}
        else:
            draw = int(digest(f"{seed}:{group}".encode("utf-8"))[:16], 16) / 2**64
            split = "test" if draw < test_ratio else "validation" if draw < test_ratio + validation_ratio else "train"
        source, info = row["source"], row["info"]
        record = {
            "id": digest((info["pixel_sha256"] + "\0" + normalized(row["caption"])).encode("utf-8"))[:24],
            "image": str(row["path"]), "caption": row["caption"],
            "source_text": source.get("source_text") or "", "isbn": str(source.get("isbn") or ""),
            "title": str(source.get("title") or ""), "group": group, "split": split,
            "image_sha256": info["image_sha256"], "width": info["width"], "height": info["height"],
        }
        split_rows[split].append(record)
    groups_by_split = {split: {r["group"] for r in rows} for split, rows in split_rows.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups_by_split[a] & groups_by_split[b]:
            raise RuntimeError("Internal group leakage check failed")
    report = {
        "manifest": str(manifest.resolve()), "image_root": str(image_root),
        "existing_split_dir": str(existing_split_dir.resolve()) if existing_split_dir is not None else None,
        "output_dir": str(output_dir.resolve()),
        "partial_preview": limit is not None, "limit": limit, "workers": workers, "seed": seed,
        "training_task": "caption condition -> corresponding image target; source_text is metadata, not a grounding filter",
        "counts": {"manifest_rows_selected": len(selected), "image_paths_checked": len(unique_paths),
                   "decoded_rows": sum("error" not in inspected[row["path"]] for row in candidates),
                   "kept_rows": len(kept), "excluded_rows": len({row["row"] for row in exclusions}),
                   "split_rows": {key: len(rows) for key, rows in split_rows.items()},
                   "split_groups": {key: len(groups) for key, groups in groups_by_split.items()},
                   "conflicting_image_groups": len(conflicts), "same_caption_duplicates_removed": duplicate_count,
                   "exif_transposed_unique_images": sum(bool(info.get("exif_transposed")) for info in inspected.values()),
                   "caption_overrides_applied": override_applied, "empty_source_text_kept": sum(not r["source_text"] for rows in split_rows.values() for r in rows)},
        "existing_rows_indexed": dict(existing_counts), "inherited_split_rows": dict(inherited),
        "existing_split_conflicts": mixed_existing, "exclusion_counts": dict(Counter(r["reason"] for r in exclusions)),
        "exclusions": exclusions, "caption_conflicts": conflicts,
        "policies": {"group_links": "ISBN, normalized title, normalized context, existing Qwen group, encoded-image SHA256 and EXIF-oriented RGB pixel SHA256",
                     "existing_split_precedence": "test > validation > train; inherited assignments override random ratios",
                     "unseen_group_ratios": {"train": 1-validation_ratio-test_ratio, "validation": validation_ratio, "test": test_ratio},
                     "same_image_caption_conflict": "Exclude every row; normalized whitespace/NFKC equality only",
                     "duplicate_image_same_caption": "Keep first manifest row; retain all linked book identities in split grouping"},
        "warnings": ["Structural image-caption candidates, not human-verified semantic-clean pairs.",
                     "Caption-image semantic agreement remains uncertain; source_text disagreement is not filtered for SANA image training.",
                     "Exact encoded/pixel hashes do not detect all resized, recompressed, or near-duplicate images.",
                     "Existing split conflicts are promoted conservatively; this does not undo prior LLM exposure to a held-out book.",
                     "EXIF is checked without changing image files; training must also apply ImageOps.exif_transpose."]
                    + (["PARTIAL PREVIEW: remaining manifest images and conflicts were not checked; do not use this report as full-data clearance."] if limit is not None else [])
                    + (["NO EXISTING SPLITS: independent new book-group assignments; not compatible with previous LLM evaluation splits unless book boundaries are explicitly coordinated."] if existing_split_dir is None else [])
                    + (["Caption overrides cover only part of this selection; inspect output language consistency."] if 0 < override_applied < len(selected) else []),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    for split, rows in split_rows.items():
        with (output_dir / (split + ".jsonl")).open(mode, encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "report.json").open(mode, encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=WORKSPACE / "검수 결과" / "clean_manifest.jsonl")
    parser.add_argument("--image-root", type=Path, default=WORKSPACE, help="Root for relative image paths in manifests/overrides")
    split_options = parser.add_mutually_exclusive_group()
    split_options.add_argument("--existing-split-dir", type=Path, default=WORKSPACE / "qwen_qa_fresh_v1" / "data")
    split_options.add_argument("--no-existing-splits", action="store_true",
                               help="Explicit independent new experiment; do not inherit previous Qwen splits")
    parser.add_argument("--output-dir", type=Path, default=HERE / "data")
    parser.add_argument("--caption-overrides", type=Path, help="Optional JSONL with image and caption fields; keyed by resolved image path")
    parser.add_argument("--limit", type=int, help="Partial development preview only; default processes all manifest rows")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--validation-ratio", type=float, default=.1)
    parser.add_argument("--test-ratio", type=float, default=.1)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace only generated split/report files")
    args = parser.parse_args()
    no_existing_splits = vars(args).pop("no_existing_splits")
    if no_existing_splits:
        args.existing_split_dir = None
    result = prepare(**vars(args))
    print(json.dumps({"partial_preview": result["partial_preview"], "counts": result["counts"],
                      "output_dir": result["output_dir"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
