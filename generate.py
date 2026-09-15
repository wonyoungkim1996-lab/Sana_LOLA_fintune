"""Generate images from English captions, using the earlier SANA 4.8B setup.

Accepts a prompt, one paired caption JSON, or LLM-produced JSONL. Adapter
inference inherits the exact model revision, prefix and resolution from training.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import time

MODEL = "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"
REVISION = "9468102c3cebb657f8c4b5f1e5a71e989a15f10d"
ASIAN_SCRIPT = re.compile(r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af\u3040-\u30ff\u3400-\u9fff]")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,150}\Z")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unique_object(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"Duplicate JSON key: {key}")
        output[key] = value
    return output


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_caption(caption):
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("imageCaption must be a nonempty string")
    if ASIAN_SCRIPT.search(caption) or not re.search(r"[A-Za-z]", caption):
        raise ValueError("Supply the English caption; this script does not translate")
    if re.search(r"```|</?think>|<\|", caption):
        raise ValueError("Caption contains model control or formatting tokens")
    return caption


def read_prompts(args):
    if args.prompt is not None:
        rows = [{"id": "prompt_001", "imageCaption": args.prompt}]
    elif args.caption_json is not None:
        data = read_json(args.caption_json)
        if not isinstance(data, dict) or set(data) != {"imageCaption"}:
            raise ValueError("Paired JSON must contain exactly imageCaption")
        rows = [{"id": args.caption_json.stem, **data}]
    else:
        rows = [json.loads(line, object_pairs_hook=unique_object)
                for line in args.prompts_jsonl.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No input captions")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "imageCaption"}:
            raise ValueError("Prompt JSONL rows require exactly id and imageCaption")
        identity = row["id"]
        if not isinstance(identity, str) or not SAFE_ID.fullmatch(identity):
            raise ValueError("Caption id must be a safe ASCII filename stem")
        if identity.casefold() in seen:
            raise ValueError(f"Duplicate caption id: {identity}")
        seen.add(identity.casefold())
        validate_caption(row["imageCaption"])
    return rows


def resolve_settings(args):
    trained = None
    if args.adapter:
        trained = read_json(args.adapter / "config.json")
        if not all(k in trained for k in ("model_id", "revision", "cache_config")):
            raise ValueError("Adapter requires this package's training config.json")
        if not (args.adapter / "pytorch_lora_weights.safetensors").is_file():
            raise FileNotFoundError("Adapter weights are missing")
        if args.model_id and args.model_id != trained["model_id"]:
            raise ValueError("Requested base model differs from adapter training")
        if args.revision and args.revision != trained["revision"]:
            raise ValueError("Requested model revision differs from adapter training")
    model_id = trained["model_id"] if trained else args.model_id or MODEL
    revision = trained["revision"] if trained else args.revision or (REVISION if model_id == MODEL else None)
    condition = trained["cache_config"] if trained else {}
    if trained and condition.get("caption_language") != "en":
        raise ValueError("This English pipeline requires an English-trained adapter")
    resolution = args.resolution or condition.get("resolution", 1024)
    if trained and resolution != condition["resolution"]:
        raise ValueError("Use the training resolution for this controlled comparison")
    prefix = condition.get("prefix", "")
    limit = condition.get("max_sequence_length", 300)
    if resolution < 32 or resolution % 32:
        raise ValueError("Resolution must be a positive multiple of 32")
    if revision and revision.startswith("local:"):
        if not Path(model_id).is_dir():
            raise ValueError("Local-model adapter requires its original local model directory")
    return {"model_id": model_id, "revision": revision, "resolution": resolution,
            "prefix": prefix, "max_sequence_length": limit,
            "caption_language": "en", "complex_human_instruction": None,
            "clean_caption": False, "use_resolution_binning": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--caption-json", type=Path)
    source.add_argument("--prompts-jsonl", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-cpu-offload", action="store_true")
    parser.add_argument("--check-input", action="store_true", help="Validate input/config only; no weights, GPU or network")
    args = parser.parse_args()
    if args.steps < 1 or not 0 <= args.seed < 2**63 or not 0 <= args.guidance_scale < 100:
        parser.error("Invalid steps, seed or guidance")
    rows, settings = read_prompts(args), resolve_settings(args)
    if args.seed + len(rows) - 1 >= 2**63:
        raise ValueError("Caption seed sequence is out of range")
    if args.check_input:
        print(json.dumps({"status": "input_valid", "captions": len(rows), **settings,
                          "token_lengths_checked": False, "model_executed": False}, ensure_ascii=True))
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve previous images")
    from huggingface_hub import snapshot_download
    import torch
    from diffusers import AutoencoderDC, SanaPipeline
    from transformers import AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA GPU with BF16 support is required")
    torch.set_num_threads(4)
    if Path(settings["model_id"]).is_dir():
        snapshot = Path(settings["model_id"]).resolve()
        local_hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.safetensors"))}
        local_hashes.update({str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*.json"))})
        actual = "local:" + hashlib.sha256(json.dumps(local_hashes, sort_keys=True).encode()).hexdigest()
        if settings["revision"] and settings["revision"] != actual:
            raise ValueError("Local model content differs from the training revision")
        settings["revision"] = actual
    else:
        snapshot = Path(snapshot_download(settings["model_id"], revision=settings["revision"],
                           local_files_only=args.local_files_only,
                           allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*"]))
        settings["revision"] = snapshot.name
    tokenizer = AutoTokenizer.from_pretrained(snapshot, subfolder="tokenizer", local_files_only=True)
    prompts = []
    for index, row in enumerate(rows):
        prompt = settings["prefix"] + row["imageCaption"]
        count = len(tokenizer(prompt.lower().strip(), truncation=False, add_special_tokens=True)["input_ids"])
        if count > settings["max_sequence_length"]:
            raise ValueError(f"{row['id']}: {count} tokens exceeds {settings['max_sequence_length']}; no silent truncation")
        prompts.append({**row, "prompt": prompt, "tokens": count, "seed": args.seed + index})
    del tokenizer
    output.mkdir(parents=True)
    atomic_json(output / "run_config.json", {**settings, "steps": args.steps,
        "guidance_scale": args.guidance_scale, "cpu_offload": not args.no_cpu_offload,
        "dtype": "bfloat16", "vae_dtype": "float32", "adapter": str(args.adapter) if args.adapter else None,
        "adapter_sha256": digest(args.adapter / "pytorch_lora_weights.safetensors") if args.adapter else None,
        "script_sha256": digest(__file__), "training": False, "total_images": len(rows)})
    atomic_json(output / "prompt_audit.json", prompts)
    completed = []
    try:
        vae = AutoencoderDC.from_pretrained(snapshot, subfolder="vae", torch_dtype=torch.float32,
                                           local_files_only=True, use_safetensors=True)
        pipe = SanaPipeline.from_pretrained(snapshot, vae=vae, torch_dtype=torch.bfloat16,
                                            local_files_only=True, use_safetensors=True)
        del vae
        if args.adapter:
            pipe.load_lora_weights(str(args.adapter), adapter_name="storybook")
        if args.no_cpu_offload:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        with (output / "outputs.jsonl").open("w", encoding="utf-8") as ledger:
            for row in prompts:
                started = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    result = pipe(prompt=row["prompt"], height=settings["resolution"], width=settings["resolution"],
                        num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(row["seed"]),
                        max_sequence_length=settings["max_sequence_length"], complex_human_instruction=None,
                        clean_caption=False, use_resolution_binning=False)
                target = output / (row["id"] + ".png")
                temporary = target.with_suffix(".tmp.png")
                result.images[0].save(temporary)
                temporary.replace(target)
                item = {**row, "image": target.name, "sha256": digest(target),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
                ledger.write(json.dumps(item, ensure_ascii=False) + "\n")
                ledger.flush()
                completed.append(item)
                atomic_json(output / "status.json", {"status": "running", "completed": len(completed), "total": len(rows)})
                print(json.dumps({"id": row["id"], "saved": str(target), "seconds": item["elapsed_seconds"]}), flush=True)
        gallery = ["<!doctype html><meta charset='utf-8'><title>SANA English captions</title>",
                   "<style>body{font-family:system-ui;max-width:1000px;margin:2rem auto}img{max-width:100%;height:auto}</style>"]
        for row in completed:
            gallery.append(f"<p>{html.escape(row['imageCaption'])}</p><img src='{row['image']}' loading='lazy'>")
        (output / "index.html").write_text("\n".join(gallery), encoding="utf-8")
        atomic_json(output / "status.json", {"status": "complete", "completed": len(completed), "total": len(rows)})
    except BaseException as exc:
        atomic_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc), "completed": len(completed)})
        raise


if __name__ == "__main__":
    main()
