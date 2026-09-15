"""Create a source-only ZIP using an explicit file allowlist (no data/weights)."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
FILES = ["README.md", "VALIDATION.md", "LICENSE", "NOTICE", ".gitignore",
         "requirements.txt", "prepare_data.py", "cache_features.py", "train_lora.py",
         "sample_compare.py", "preflight.py", "package_code.py",
         "test_prepare_data.py", "test_sample_compare.py", "test_training_integration.py",
         "test_cache_features.py", "build_manifest.py", "test_build_manifest.py",
         "prepare_pairs.py", "test_prepare_pairs.py", "image_io.py", "test_image_io.py",
         "generate.py", "test_generate.py", "run_smoke.py", "examples/prompts.jsonl",
         "docs/PREVIOUS_EXPERIMENTS.md", "docs/teacher_lora_smoke_validation.json"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "Sana_LOLA_fintune_source.zip")
    args = parser.parse_args()
    missing = [name for name in FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete source package: {missing}")
    hashes = {}
    with ZipFile(args.output, "w", ZIP_DEFLATED) as archive:
        for name in FILES:
            payload = (ROOT / name).read_bytes()
            archive.writestr(f"Sana_LOLA_fintune/{name}", payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        archive.writestr("Sana_LOLA_fintune/source_sha256.json", json.dumps(hashes, indent=2))
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed.")
    print(json.dumps({"zip": str(args.output.resolve()), "source_files": len(FILES),
                      "bytes": args.output.stat().st_size, "data_or_weights_included": False}, indent=2))


if __name__ == "__main__":
    main()
