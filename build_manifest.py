"""Build a portable SANA caption/image manifest from the original label JSONs.

Supported schema: root title/isbn and imageInfo[] containing srcImageFile,
optional srcImagePath/srcText, and imageCaptionInfo.imageCaption. No captions
are generated, no image is copied, and no fuzzy filename matching is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath, PureWindowsPath


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def contained(path, root):
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return False


def resolve_image(filename, source_path, images_dir, by_name):
    """Only a valid referenced path or a unique exact basename can resolve."""
    if not isinstance(filename, str) or not filename.strip():
        return None, "missing_image_reference"
    reference = filename.strip().replace("\\", "/")
    file_part = PurePosixPath(reference)
    windows_absolute = bool(PureWindowsPath(reference).drive)
    candidates = set()
    local = Path(reference)
    if local.is_absolute() or windows_absolute:
        if local.is_absolute() and local.is_file() and contained(local, images_dir):
            candidates.add(local.resolve())
    elif len(file_part.parts) > 1:
        direct = images_dir / local
        if direct.is_file() and contained(direct, images_dir):
            candidates.add(direct.resolve())
    if isinstance(source_path, str) and source_path.strip():
        source = Path(source_path.strip().replace("\\", "/"))
        combined = source / file_part.name if source.is_absolute() else images_dir / source / file_part.name
        if combined.is_file() and contained(combined, images_dir):
            candidates.add(combined.resolve())
    if len(candidates) == 1:
        return next(iter(candidates)), "referenced_path"
    if len(candidates) > 1:
        return None, "ambiguous_referenced_paths"
    # Case is folded for Windows portability, but substrings/stems are never used.
    candidates = by_name.get(file_part.name.casefold(), [])
    if len(candidates) == 1:
        return candidates[0], "unique_exact_basename"
    return None, "ambiguous_exact_basename" if candidates else "image_not_found"


def build_manifest(labels_dir, images_dir, output, *, overwrite=False, limit_labels=None):
    labels_dir, images_dir, output = map(lambda value: Path(value).resolve(), (labels_dir, images_dir, output))
    report_path = output.with_suffix(".report.json")
    if not labels_dir.is_dir() or not images_dir.is_dir():
        raise FileNotFoundError("--labels-dir and --images-dir must be existing directories")
    if output.suffix.lower() != ".jsonl":
        raise ValueError("--output must end in .jsonl")
    if limit_labels is not None and limit_labels < 1:
        raise ValueError("limit_labels must be positive")
    if not overwrite and (output.exists() or report_path.exists()):
        raise FileExistsError("Manifest/report exists; choose a new output or use --overwrite")
    if contained(report_path, labels_dir):
        raise ValueError("Write generated manifest/report outside the original labels directory")
    by_name = defaultdict(list)
    image_files = []
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and contained(path, images_dir):
            resolved = path.resolve()
            image_files.append(resolved)
            by_name[path.name.casefold()].append(resolved)
    for name, paths in by_name.items():
        by_name[name] = sorted(set(paths))
    label_files = sorted(path for path in labels_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".json")
    total_label_files = len(label_files)
    if limit_labels is not None:
        label_files = label_files[:limit_labels]
    rows, exclusions = [], []
    resolution_counts = Counter()
    entries_seen = 0
    for label in label_files:
        try:
            payload = json.loads(label.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, ValueError) as exc:
            exclusions.append({"label_json": str(label), "reason": "invalid_label_json", "detail": str(exc)})
            continue
        entries = payload.get("imageInfo") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not entries:
            exclusions.append({"label_json": str(label), "reason": "missing_or_invalid_imageInfo"})
            continue
        for index, info in enumerate(entries):
            entries_seen += 1
            where = {"label_json": str(label), "image_info_index": index}
            if not isinstance(info, dict):
                exclusions.append(dict(where, reason="invalid_imageInfo_entry"))
                continue
            caption_info = info.get("imageCaptionInfo")
            caption = caption_info.get("imageCaption") if isinstance(caption_info, dict) else None
            if not isinstance(caption, str) or not caption.strip():
                exclusions.append(dict(where, reason="empty_or_missing_caption"))
                continue
            image, resolution = resolve_image(info.get("srcImageFile"), info.get("srcImagePath"), images_dir, by_name)
            if image is None:
                exclusions.append(dict(where, reason=resolution, srcImageFile=info.get("srcImageFile")))
                continue
            resolution_counts[resolution] += 1
            rows.append({
                "image": str(image), "caption": caption,
                "source_text": info.get("srcText") if isinstance(info.get("srcText"), str) else "",
                "isbn": str(payload.get("isbn") or ""), "title": str(payload.get("title") or ""),
                "label_json": str(label), "src_text_id": str(info.get("srcTextID") or ""),
                "image_info_index": index,
            })
    report = {
        "labels_dir": str(labels_dir), "images_dir": str(images_dir), "manifest": str(output),
        "schema": "imageInfo[].imageCaptionInfo.imageCaption paired by referenced imageInfo[].srcImageFile",
        "partial_preview": limit_labels is not None, "limit_labels": limit_labels,
        "counts": {"label_files_available": total_label_files, "label_files_read": len(label_files),
                   "image_files_indexed": len(image_files), "image_info_entries_seen": entries_seen,
                   "written_rows": len(rows), "exclusion_events": len(exclusions),
                   "empty_source_text_rows": sum(not row["source_text"] for row in rows)},
        "resolution_counts": dict(resolution_counts),
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)), "exclusions": exclusions,
        "warnings": [
            "References and nonempty captions were checked; image decoding, hashes, EXIF, conflicts and split grouping belong to prepare_data.py.",
            "Caption-image semantic correctness is unverified; no captions are invented or filtered against source_text.",
            "Unique exact basename fallback supports stale source directory metadata; ambiguous names are excluded without fuzzy matching.",
            "Duplicate records are retained here so prepare_data.py can group all book identities before deduplication.",
        ] + (["PARTIAL PREVIEW: only the requested first label files were read."] if limit_labels is not None else []),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with output.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with report_path.open(mode, encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, required=True, help="Original label JSON root, searched recursively")
    parser.add_argument("--images-dir", type=Path, required=True, help="Original target image root, searched recursively")
    parser.add_argument("--output", type=Path, required=True, help="Generated .jsonl manifest outside the label root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-labels", type=int, help="Partial CPU development preview; default reads all label files")
    args = parser.parse_args()
    result = build_manifest(**vars(args))
    print(json.dumps({"counts": result["counts"], "exclusion_counts": result["exclusion_counts"],
                      "partial_preview": result["partial_preview"], "manifest": result["manifest"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
