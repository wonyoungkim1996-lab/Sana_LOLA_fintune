"""Cache caption embeddings and paired image latents in separate GPU stages.

No SANA training is launched. Caption padding is omitted on disk. Test data is
not cached by default. The original image/label files are never modified.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"
DEFAULT_REVISION = "9468102c3cebb657f8c4b5f1e5a71e989a15f10d"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_pixels(path, resolution, expected_sha256=None):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from image_io import load_rgb
    with load_rgb(path, expected_sha256=expected_sha256) as original:
        fitted = ImageOps.contain(original, (resolution, resolution), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (resolution, resolution), "white")
        canvas.paste(fitted, ((resolution - fitted.width) // 2, (resolution - fitted.height) // 2))
        pixels = np.asarray(canvas, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def trim_embeddings(embeddings, mask):
    visible = mask.nonzero(as_tuple=False)
    end = int(visible[-1].item()) + 1 if len(visible) else 1
    return {"prompt_embeds": embeddings[:end].detach().cpu().contiguous(),
            "prompt_attention_mask": mask[:end].detach().cpu().contiguous()}


def valid_text_cache(path, fingerprint, identity):
    """Resume only a complete, finite tensor file for this exact caption cache."""
    import torch
    from safetensors import safe_open
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if handle.metadata() != {"fingerprint": fingerprint, "id": identity}:
                return False
            if set(handle.keys()) != {"prompt_embeds", "prompt_attention_mask"}:
                return False
            embeds = handle.get_tensor("prompt_embeds")
            mask = handle.get_tensor("prompt_attention_mask")
            return bool(embeds.ndim == 2 and mask.ndim == 1 and embeds.shape[0] == len(mask)
                        and len(mask) > 0 and torch.isfinite(embeds).all()
                        and ((mask == 0) | (mask == 1)).all() and mask.any())
    except Exception:
        return False


def read_journal(path):
    if not path.exists():
        return {}
    rows = {}
    lines = path.read_bytes().splitlines()
    valid = []
    for index, raw_line in enumerate(lines):
        try:
            line = raw_line.decode("utf-8")
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise
            break
        rows[row["id"]] = row
        valid.append(line)
    # Preserve the valid prefix after an interrupted last write.
    if len(valid) != len(lines):
        backup = path.with_suffix(".interrupted.jsonl")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_bytes(("\n".join(valid) + ("\n" if valid else "")).encode("utf-8"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--model-id", default=MODEL)
    parser.add_argument("--revision", help="Defaults to the previous smoke model revision only for the default model ID.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=300)
    parser.add_argument("--prefix", default="", help="Optional fixed prefix; default preserves caption-only conditioning.")
    parser.add_argument("--caption-language", choices=["en"], default="en")
    parser.add_argument("--splits", nargs="+", choices=["train", "validation", "test"], default=["train", "validation"])
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--min-free-gib", type=float, default=6.0,
                        help="CUDA free-memory preflight floor; does not guarantee the model fits.")
    args = parser.parse_args()
    if args.revision is None and args.model_id == MODEL:
        args.revision = DEFAULT_REVISION
    if args.resolution < 32 or args.resolution % 32 or args.max_sequence_length < 1:
        parser.error("Resolution must be a positive multiple of 32 and sequence length positive.")
    if any(value is not None and value < 1 for value in [args.train_limit, args.validation_limit]):
        parser.error("Limits must be positive; omit a limit to use all rows.")
    records, data_hashes, source_counts = [], {}, {}
    for split in dict.fromkeys(args.splits):
        path = args.data_dir / f"{split}.jsonl"
        data_hashes[split] = sha256(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row["split"] != split for row in rows):
            raise ValueError(f"Wrong split value in {path}")
        import re
        for row in rows:
            caption = row.get("caption")
            if not isinstance(caption, str) or not re.search(r"[A-Za-z]", caption) or re.search(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3]", caption):
                raise ValueError(f"Expected an English caption without Hangul in {path}: {row.get('id')}")
        rows.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
        source_counts[split] = len(rows)
        limit = args.train_limit if split == "train" else args.validation_limit if split == "validation" else None
        records.extend(rows[:limit] if limit is not None else rows)
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Cache requires nonempty records with unique IDs.")
    groups, hashes = {}, {}
    for row in records:
        for key, mapping in [(row["group"], groups), (row["image_sha256"], hashes)]:
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("Book or image leakage between selected splits.")
            mapping[key] = row["split"]
    config = {"resolution": args.resolution, "max_sequence_length": args.max_sequence_length,
              "prefix": args.prefix, "caption_language": args.caption_language, "clean_caption": False,
              "encoder_compute_dtype": "bfloat16" if args.device == "cuda" else "float32",
              "vae_compute_dtype": "float32",
              "complex_human_instruction": None,
              "preprocessing": "exif_transpose_alpha_on_white_rgb_white_letterbox_lanczos_normalize_minus1_plus1_v2",
              "image_io_sha256": sha256(ROOT / "image_io.py"),
              "cache_schema_version": 2,
              "latent_scaling": "vae.encode(...).latent * vae.config.scaling_factor",
              "embedding_storage": "trimmed_to_last_attended_token"}

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if Path(args.model_id).is_dir():
        snapshot = Path(args.model_id).resolve()
        local_hashes = {str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob("*.safetensors"))}
        local_hashes.update({str(p.relative_to(snapshot)): sha256(p) for p in sorted(snapshot.rglob("*.json"))})
        revision = "local:" + hashlib.sha256(json.dumps(local_hashes, sort_keys=True).encode()).hexdigest()
    else:
        snapshot = Path(snapshot_download(args.model_id, revision=args.revision,
                                         local_files_only=args.local_files_only,
                                         allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*",
                                                         "text_encoder/*", "transformer/*", "vae/*"]))
        revision = snapshot.name
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer", local_files_only=True)
    token_counts, too_long = {}, []
    for row in records:
        prompt = (args.prefix + row["caption"]).lower().strip()
        count = len(tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"])
        token_counts[row["id"]] = count
        if count > args.max_sequence_length:
            too_long.append({"id": row["id"], "tokens": count})
    specification = {"model_id": args.model_id, "revision": revision, "config": config,
                     "data_sha256": data_hashes, "selected_ids": [r["id"] for r in records]}
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = args.cache_dir.resolve() / fingerprint[:16]
    cache.mkdir(parents=True, exist_ok=True)
    if too_long:
        write_json(cache / "overlength.json", too_long)
        raise ValueError(f"{len(too_long)} captions exceed the token limit. No caption was silently truncated. See {cache / 'overlength.json'}")
    metadata = {**specification, "fingerprint": fingerprint, "snapshot_path": str(snapshot),
                "source_counts": source_counts,
                "selected_counts": {s: sum(r["split"] == s for r in records) for s in source_counts},
                "partial": any(sum(r["split"] == s for r in records) < n for s, n in source_counts.items()),
                "status": "building", "records": []}
    existing_path = cache / "cache_manifest.json"
    if existing_path.exists():
        old = json.loads(existing_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Cache fingerprint collision; select another cache directory.")
    transformer_config = json.loads((snapshot / "transformer/config.json").read_text())
    vae_config = json.loads((snapshot / "vae/config.json").read_text())
    if float(vae_config.get("shift_factor") or 0.0) != 0.0:
        raise ValueError("This SANA AutoencoderDC cache supports scaling-only latents; nonzero VAE shift is unsupported.")
    metadata["vae_scaling_factor"] = vae_config["scaling_factor"]
    caption_channels = transformer_config["caption_channels"]
    estimated_bytes = sum((token_counts[r["id"]] * caption_channels * 2 + 32 * (args.resolution // 32)**2 * 2 + 8192) for r in records)
    metadata["estimated_cache_gib"] = estimated_bytes / 2**30
    journal = cache / "feature_journal.jsonl"
    completed = read_journal(journal)
    for key, row in list(completed.items()):
        path = cache / row["feature_file"]
        if not path.is_file() or sha256(path) != row["feature_sha256"]:
            del completed[key]
    outstanding = [row for row in records if row["id"] not in completed]
    required = estimated_bytes * len(outstanding) / len(records) + 512 * 2**20
    if shutil.disk_usage(cache).free < required:
        raise OSError(f"Insufficient disk space; estimated total cache is {estimated_bytes / 2**30:.2f} GiB. Use a smaller pilot or another --cache-dir.")
    write_json(existing_path, metadata)
    print(json.dumps({"stage": "cache_plan", "cache": str(cache), "selected_counts": metadata["selected_counts"],
                      "remaining": len(outstanding), "estimated_cache_gib": metadata["estimated_cache_gib"]}), flush=True)

    import torch
    from diffusers import AutoencoderDC, SanaPipeline
    from safetensors.torch import load_file, save_file
    from transformers import Gemma2Model

    torch.set_num_threads(4)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. No model was loaded onto GPU.")
    def check_gpu(stage):
        if args.device == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This CUDA cache configuration requires BF16 support.")
            free = torch.cuda.mem_get_info()[0] / 2**30
            if free < args.min_free_gib:
                raise RuntimeError(f"{stage}: free CUDA memory {free:.2f} GiB is below --min-free-gib {args.min_free_gib}.")
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    text_directory = cache / "text"
    feature_directory = cache / "features"
    text_directory.mkdir(exist_ok=True)
    feature_directory.mkdir(exist_ok=True)
    negative_path = cache / "negative_prompt.safetensors"

    def save_tensors(path, values, identity):
        if any(not torch.isfinite(tensor).all() for tensor in values.values()):
            raise ValueError(f"Non-finite cached tensor: {identity}")
        temporary = path.with_suffix(".tmp")
        save_file(values, str(temporary), metadata={"fingerprint": fingerprint, "id": identity})
        temporary.replace(path)

    need_text = [row for row in outstanding
                 if not valid_text_cache(text_directory / f"{row['id']}.safetensors", fingerprint, row["id"])]
    negative_valid = valid_text_cache(negative_path, fingerprint, "negative")
    if need_text or not negative_valid:
        check_gpu("text_encoder")
        encoder = Gemma2Model.from_pretrained(snapshot, subfolder="text_encoder", torch_dtype=dtype, local_files_only=True)
        encoder.requires_grad_(False).eval().to(args.device)
        text_pipeline = SanaPipeline(tokenizer=tokenizer, text_encoder=encoder, vae=None, transformer=None, scheduler=None)

        def encode(prompt):
            with torch.no_grad():
                embeddings, mask, _, _ = text_pipeline.encode_prompt(prompt, do_classifier_free_guidance=False,
                    device=torch.device(args.device), max_sequence_length=args.max_sequence_length,
                    clean_caption=False, complex_human_instruction=None)
            return trim_embeddings(embeddings[0].to(torch.bfloat16), mask[0])

        if not negative_valid:
            save_tensors(negative_path, encode(""), "negative")
        for index, row in enumerate(need_text, 1):
            save_tensors(text_directory / f"{row['id']}.safetensors", encode(args.prefix + row["caption"]), row["id"])
            if index % 10 == 0 or index == len(need_text):
                print(json.dumps({"stage": "text_cache", "done": index, "total": len(need_text)}), flush=True)
        del text_pipeline, encoder
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    del tokenizer
    if outstanding:
        check_gpu("vae")
        vae = AutoencoderDC.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32, local_files_only=True)
        vae.requires_grad_(False).eval().to(args.device)
        if vae.config.scaling_factor != metadata["vae_scaling_factor"]:
            raise ValueError("Loaded VAE scaling factor differs from the snapshot config.")
        with journal.open("a", encoding="utf-8") as log:
            for index, row in enumerate(outstanding, 1):
                text_path = text_directory / f"{row['id']}.safetensors"
                values = load_file(str(text_path))
                with torch.no_grad():
                    latent = vae.encode(prepare_pixels(row["image"], args.resolution,
                        expected_sha256=row["image_sha256"]).to(args.device)).latent
                    values["latents"] = (latent[0] * vae.config.scaling_factor).to("cpu", torch.bfloat16).contiguous()
                destination = feature_directory / f"{row['id']}.safetensors"
                save_tensors(destination, values, row["id"])
                result = {**row, "feature_file": str(destination.relative_to(cache)), "feature_sha256": sha256(destination),
                          "tokens": token_counts[row["id"]]}
                log.write(json.dumps(result, ensure_ascii=False) + "\n")
                log.flush()
                completed[row["id"]] = result
                text_path.unlink()
                if index % 10 == 0 or index == len(outstanding):
                    print(json.dumps({"stage": "image_cache", "done": index, "total": len(outstanding)}), flush=True)
        del vae
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    metadata.update(status="complete", records=[completed[r["id"]] for r in records],
                    negative_feature_file=negative_path.name, negative_feature_sha256=sha256(negative_path))
    write_json(existing_path, metadata)
    write_json(args.cache_dir / "latest.json", {"cache_path": str(cache), "fingerprint": fingerprint,
                                               "status": "complete", "partial": metadata["partial"]})
    print(json.dumps({"stage": "cache_complete", "cache": str(cache), "records": len(records)}), flush=True)


if __name__ == "__main__":
    main()
