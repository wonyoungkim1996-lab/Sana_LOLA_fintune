"""Generate paired base/LoRA images from the exact cached validation captions.

This compares outputs; it does not claim automated semantic accuracy. No external
API is used. The text encoder is omitted because embeddings were cached separately.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_cache(path):
    path = Path(path).resolve()
    if path.is_file():
        pointer = read_json(path)
        if "cache_path" in pointer:
            candidate = Path(pointer["cache_path"])
            path = candidate if candidate.is_absolute() else path.parent / candidate
        else:
            path = path.parent
    return path, read_json(path / "cache_manifest.json")


def select_records(records, split, count):
    """Prefer different books; deterministic and independent of model outputs."""
    available = [r for r in records if r["split"] == split]
    available.sort(key=lambda r: hashlib.sha256(str(r["id"]).encode()).hexdigest())
    first, remaining, seen = [], [], set()
    for row in available:
        group = row["group"]
        if group in seen:
            remaining.append(row)
        else:
            first.append(row)
            seen.add(group)
    return (first + remaining)[:count]


def resolve_feature(cache, row, key="feature_file", hash_key="feature_sha256"):
    relative = Path(row[key])
    path = relative if relative.is_absolute() else cache / relative
    if row.get(hash_key) != digest(path):
        raise ValueError(f"Feature hash mismatch: {path}")
    return path


def validate_adapter(adapter, metadata):
    """A LoRA trained for another cache/model must not be silently compared."""
    candidates = [adapter / "config.json", adapter.parent / "config.json"]
    candidates += [adapter / "run_config.json", adapter.parent / "run_config.json"]
    for path in candidates:
        if path.exists():
            config = read_json(path)
            fingerprint = config.get("cache_fingerprint")
            if fingerprint:
                if fingerprint != metadata["fingerprint"]:
                    # A final-test cache can contain different rows but must use the
                    # identical model and conditioning/preprocessing configuration.
                    manifest_path = config.get("cache_manifest_path")
                    if not manifest_path or not Path(manifest_path).is_file():
                        raise ValueError("Different evaluation cache: training cache manifest is required to verify compatibility.")
                    if config.get("cache_manifest_sha256") != digest(manifest_path):
                        raise ValueError("The recorded training cache manifest changed.")
                    trained = read_json(manifest_path)
                    if trained.get("fingerprint") != fingerprint:
                        raise ValueError("Training manifest fingerprint differs from the adapter configuration.")
                    if any(trained.get(key) != metadata.get(key) for key in ["model_id", "revision", "config"]):
                        raise ValueError("Adapter model or cache preprocessing differs from evaluation.")
                    train_rows = [r for r in trained["records"] if r["split"] == "train"]
                    evaluation_rows = [r for r in metadata["records"] if r["split"] in ["validation", "test"]]
                    train_groups = {r["group"] for r in train_rows}
                    train_hashes = {r.get("image_sha256") for r in train_rows} - {None}
                    if any(r["group"] in train_groups or r.get("image_sha256") in train_hashes for r in evaluation_rows):
                        raise ValueError("Evaluation rows overlap the adapter's training books/images.")
                return path
    raise ValueError("Cannot find the training config with cache_fingerprint beside the adapter.")


def guidance_embeddings(positive, negative, device, dtype):
    """Pad cached, masked sequences to one CFG length without losing text tokens."""
    import torch.nn.functional as F

    length = max(positive["prompt_embeds"].shape[0], negative["prompt_embeds"].shape[0])
    output = {}
    for prefix, values in [("", positive), ("negative_", negative)]:
        embeds, mask = values["prompt_embeds"], values["prompt_attention_mask"]
        if embeds.shape[0] != mask.shape[0]:
            raise ValueError("Cached embedding and mask lengths differ.")
        amount = length - embeds.shape[0]
        output[prefix + "prompt_embeds"] = F.pad(embeds, (0, 0, 0, amount)).unsqueeze(0).to(device=device, dtype=dtype)
        output[prefix + "prompt_attention_mask"] = F.pad(mask, (0, amount)).unsqueeze(0).to(device)
    return output


def call_cached_pipeline(pipe, embeddings, **sampling_options):
    # SanaPipeline defaults negative_prompt to "". Cached negative embeddings
    # must explicitly suppress that text argument, or check_inputs rejects both.
    return pipe(negative_prompt=None, **embeddings, **sampling_options)


def write_gallery(output, selected, seeds, conditions):
    cells = ["<!doctype html><meta charset='utf-8'><title>SANA paired outputs</title>",
             "<style>body{font-family:system-ui;margin:24px;max-width:1200px}table{width:100%;table-layout:fixed}td{vertical-align:top}img{width:100%}p{white-space:pre-wrap}</style>",
             "<h1>SANA: same captions and seeds</h1><p>Visual outputs only; no semantic accuracy score is assigned.</p>"]
    for row in selected:
        cells.append(f"<h2>{html.escape(str(row['id']))}</h2><p>{html.escape(row['caption'])}</p>")
        for seed in seeds:
            cells.append(f"<p>Seed: {seed}</p><table><tr>")
            for condition in conditions:
                filename = f"{condition}/{row['id']}_{seed}.png"
                cells.append(f"<td>{condition}<br><img src='{filename}' loading='lazy'></td>")
            cells.append("</tr></table>")
    (output / "index.html").write_text("\n".join(cells), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/latest.json")
    parser.add_argument("--adapter", type=Path, help="Saved Diffusers LoRA folder; omit for base-only outputs.")
    parser.add_argument("--output", type=Path, required=True, help="New output folder (never overwritten).")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seeds", default="17,23")
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    args = parser.parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",")]
    if args.count < 1 or args.steps < 1 or not seeds or len(seeds) != len(set(seeds)):
        parser.error("Positive count/steps and unique integer seeds are required.")
    if any(s < 0 or s >= 2**63 for s in seeds):
        parser.error("Seeds must be in [0, 2**63).")
    cache, metadata = resolve_cache(args.cache)
    config = metadata["config"]
    selected = select_records(metadata["records"], args.split, args.count)
    if not selected:
        raise ValueError(f"No cached {args.split} rows. Cache this split before sampling.")
    feature_paths = {r["id"]: resolve_feature(cache, r) for r in selected}
    negative = resolve_feature(cache, metadata, "negative_feature_file", "negative_feature_sha256")
    if args.adapter:
        args.adapter = args.adapter.resolve()
        validate_adapter(args.adapter, metadata)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists. Choose a new folder: {output}")

    import torch
    from diffusers import AutoencoderDC, SanaPipeline
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run preflight.py first.")
    torch.set_num_threads(4)
    output.mkdir(parents=True)
    conditions = ["base", "lora"] if args.adapter else ["base"]
    run = {"cache_fingerprint": metadata["fingerprint"], "model_id": metadata["model_id"],
           "revision": metadata["revision"], "split": args.split,
           "selected_ids": [r["id"] for r in selected], "seeds": seeds,
           "resolution": config["resolution"], "steps": args.steps,
           "guidance_scale": args.guidance_scale, "conditions": conditions,
           "adapter": str(args.adapter) if args.adapter else None,
           "adapter_files": {p.name: digest(p) for p in args.adapter.glob("*.safetensors")} if args.adapter else {},
           "fixed_cached_prompt_embeddings": True, "use_resolution_binning": False}
    (output / "config.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        # Loading modules on CPU first avoids placing all weights in GPU at once.
        vae = AutoencoderDC.from_pretrained(metadata["snapshot_path"], subfolder="vae",
                                            torch_dtype=torch.float32, local_files_only=True)
        pipe = SanaPipeline.from_pretrained(metadata["snapshot_path"], text_encoder=None,
                                            tokenizer=None, vae=vae, torch_dtype=torch.bfloat16,
                                            local_files_only=True)
        del vae
        pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        negative_values = load_file(str(negative))
        with (output / "outputs.jsonl").open("w", encoding="utf-8") as log:
            for condition in conditions:
                (output / condition).mkdir()
                if condition == "lora":
                    pipe.load_lora_weights(str(args.adapter), adapter_name="storybook")
                for row in selected:
                    values = load_file(str(feature_paths[row["id"]]))
                    kwargs = guidance_embeddings(values, negative_values, "cuda", torch.bfloat16)
                    for seed in seeds:
                        started = time.monotonic()
                        torch.cuda.reset_peak_memory_stats()
                        result = call_cached_pipeline(pipe, kwargs, height=config["resolution"], width=config["resolution"],
                                      num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                                      generator=torch.Generator(device="cuda").manual_seed(seed),
                                      max_sequence_length=config["max_sequence_length"],
                                      complex_human_instruction=None, clean_caption=False,
                                      use_resolution_binning=False)
                        path = output / condition / f"{row['id']}_{seed}.png"
                        result.images[0].save(path)
                        record = {"condition": condition, "id": row["id"], "group": row["group"],
                                  "caption": row["caption"], "seed": seed, "image": str(path),
                                  "image_sha256": digest(path), "elapsed_seconds": time.monotonic() - started,
                                  "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
                        log.write(json.dumps(record, ensure_ascii=False) + "\n")
                        log.flush()
                        print(json.dumps({k: record[k] for k in ["condition", "id", "seed", "elapsed_seconds"]}), flush=True)
        write_gallery(output, selected, seeds, conditions)
        (output / "completed.json").write_text(json.dumps({"images": len(selected) * len(seeds) * len(conditions)}), encoding="utf-8")
    except Exception as exc:
        (output / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
