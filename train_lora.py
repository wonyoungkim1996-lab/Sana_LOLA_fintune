"""Caption-conditioned SANA LoRA from cached tensors, without a text encoder/VAE.

Default: 100 optimizer-step pilot, batch one, gradient accumulation four. Use
--epochs explicitly for complete passes over a complete training cache. Validation
is fixed-noise flow MSE, not caption correctness or perceptual image quality.

Training equations and Diffusers adapter format follow the official SANA example:
https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora_sana.py
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def resolve_cache(path):
    path = Path(path).resolve()
    if path.is_file():
        pointer = read_json(path)
        if "cache_path" in pointer:
            candidate = Path(pointer["cache_path"])
            path = candidate if candidate.is_absolute() else path.parent / candidate
        else:
            path = path.parent
    return path.resolve(), read_json(path / "cache_manifest.json")


def feature_path(cache, row):
    relative = Path(row["feature_file"])
    path = relative if relative.is_absolute() else cache / relative
    path = path.resolve()
    if not path.is_relative_to(cache.resolve()):
        raise ValueError("Feature path must remain inside its cache directory.")
    if not path.is_file() or digest(path) != row["feature_sha256"]:
        raise ValueError(f"Missing or changed cached feature: {path}")
    return path


def data_order(size, seed, epoch):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return order


def validation_seed(record_id, seed):
    return int(hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()[:15], 16)


def make_flow_batch(clean, scheduler, noise, indices):
    """Noise-clean velocity target; timestep stays on SANA's native 0..1000 scale."""
    import torch

    indices = indices.to(device="cpu", dtype=torch.long)
    sigma = scheduler.sigmas[indices].to(device=clean.device, dtype=clean.dtype)
    timestep = scheduler.timesteps[indices].to(device=clean.device, dtype=torch.float32)
    sigma = sigma.reshape(-1, *([1] * (clean.ndim - 1)))
    return (1.0 - sigma) * clean + sigma * noise, noise - clean, timestep


def adapter_digest(model):
    from peft.utils import get_peft_model_state_dict

    value = hashlib.sha256()
    for name, tensor in sorted(get_peft_model_state_dict(model).items()):
        value.update(name.encode())
        value.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def tensor_dict_digest(tensors):
    """Hash exact tensor names, shapes, dtypes and bytes, independent of file metadata."""
    import torch

    value = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        value.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def publish_final_adapter(destination, model, config):
    """Publish a complete directory atomically, or verify an identical existing one.

    An interrupted temporary directory is never treated as a published adapter.
    Existing published weights/configuration must match the actual restored model.
    """
    from diffusers import SanaPipeline
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    destination = Path(destination).resolve()
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    try:
        adapter = {name: tensor.detach().cpu().contiguous()
                   for name, tensor in get_peft_model_state_dict(model).items()}
        SanaPipeline.save_lora_weights(save_directory=temporary, transformer_lora_layers=adapter,
                                      safe_serialization=True)
        weights_name = "pytorch_lora_weights.safetensors"
        expected_hash = tensor_dict_digest(load_file(str(temporary / weights_name)))
        publication = {"format_version": 1, "config_sha256": config_digest(config),
                       "adapter_sha256": adapter_digest(model), "public_tensor_sha256": expected_hash,
                       "weights_file": weights_name, "weights_file_sha256": digest(temporary / weights_name)}
        atomic_json(temporary / "config.json", config)
        atomic_json(temporary / "publication.json", publication)
        if destination.exists():
            if not destination.is_dir() or not (destination / "config.json").is_file():
                raise ValueError("Existing final adapter is incomplete; refusing to overwrite it.")
            if read_json(destination / "config.json") != config:
                raise ValueError("Existing final adapter configuration conflicts with the restored checkpoint.")
            if not (destination / weights_name).is_file():
                raise ValueError("Existing final adapter has no complete weights file.")
            if tensor_dict_digest(load_file(str(destination / weights_name))) != expected_hash:
                raise ValueError("Existing final adapter tensors conflict with the restored checkpoint.")
            if (destination / "publication.json").exists():
                existing = read_json(destination / "publication.json")
                for key in ("config_sha256", "adapter_sha256", "public_tensor_sha256", "weights_file"):
                    if existing.get(key) != publication[key]:
                        raise ValueError("Existing final adapter publication hash conflicts with its checkpoint.")
                if existing.get("weights_file_sha256") != digest(destination / weights_name):
                    raise ValueError("Existing final adapter file changed after publication.")
            return False
        os.replace(temporary, destination)
        return True
    finally:
        if temporary.exists():
            # Delete only this invocation's private staging directory, never the final adapter.
            resolved = temporary.resolve()
            if (resolved.parent != destination.parent or resolved == destination or
                    not resolved.name.startswith(destination.name + ".tmp-")):
                raise RuntimeError("Unsafe staging-directory cleanup target")
            shutil.rmtree(resolved)


def capture_rng(device):
    import torch

    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}


def restore_rng(state, device):
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_save(destination, model, optimizer, state, config, device):
    """Atomic directory publication; raw adapter + real optimizer/RNG/order state."""
    import torch
    from diffusers import SanaPipeline
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Checkpoint exists: {destination}")
    temporary = destination.with_name(destination.name + ".tmp-" + uuid.uuid4().hex)
    temporary.mkdir(parents=True)
    adapter = {name: tensor.detach().cpu().contiguous()
               for name, tensor in get_peft_model_state_dict(model).items()}
    SanaPipeline.save_lora_weights(save_directory=temporary, transformer_lora_layers=adapter,
                                  safe_serialization=True)
    save_file(adapter, str(temporary / "adapter_training.safetensors"))
    stored = dict(state, optimizer=optimizer.state_dict(), rng=capture_rng(device),
                  config=config, adapter_sha256=adapter_digest(model))
    torch.save(stored, temporary / "training_state.pt")
    atomic_json(temporary / "config.json", config)
    atomic_json(temporary / "checkpoint.json", {"step": state["step"],
                "epoch": state["epoch"], "cursor": state["cursor"],
                "adapter_sha256": stored["adapter_sha256"]})
    os.replace(temporary, destination)


def checkpoint_restore(checkpoint, model, optimizer, config, device):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    # This pickle is a locally produced optimizer checkpoint, not a downloaded model.
    stored = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
    if stored["config"] != config:
        changed = [key for key in set(stored["config"]) | set(config)
                   if stored["config"].get(key) != config.get(key)]
        raise ValueError(f"Resume configuration differs ({changed}); repeat the original training arguments.")
    loaded = set_peft_model_state_dict(model, load_file(str(checkpoint / "adapter_training.safetensors")),
                                       adapter_name="default")
    if loaded.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {loaded.unexpected_keys}")
    if adapter_digest(model) != stored["adapter_sha256"]:
        raise RuntimeError("Adapter reload hash does not match saved adapter.")
    optimizer.load_state_dict(stored["optimizer"])
    restore_rng(stored["rng"], device)
    return {key: value for key, value in stored.items() if key not in ("optimizer", "rng", "config", "adapter_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/latest.json")
    parser.add_argument("--output", type=Path, required=True)
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--max-steps", type=int, help="Optimizer steps; default100 is a pilot, not full training.")
    duration.add_argument("--epochs", type=int, help="Explicit complete passes; requires a complete training cache.")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=1729)
    parser.add_argument("--validation-limit", type=int, default=32)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=1.0,
                        help="Explicit training shift.1 matches official FlowMatch training default; inference scheduler may differ.")
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", type=Path, help="Locally created checkpoint directory; output must be its original run.")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="CPU is for tiny synthetic implementation tests, not a speed recommendation.")
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--stop-after-steps", type=int,
                        help="Intentional checkpoint-stop for recovery tests, preserving original total-step schedule.")
    args = parser.parse_args()
    if args.max_steps is None and args.epochs is None:
        args.max_steps = 100
    for name in ("rank", "accumulation", "validation_limit", "validation_every", "checkpoint_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if any(value is not None and value < 1 for value in (args.max_steps, args.epochs, args.stop_after_steps)):
        parser.error("Step and epoch counts must be positive")
    if args.learning_rate <= 0 or args.flow_shift <= 0 or args.warmup_steps < 0 or args.max_grad_norm <= 0:
        parser.error("Invalid optimizer/scheduler settings")

    cache, metadata = resolve_cache(args.cache)
    if not metadata.get("records"):
        raise ValueError("Cache has no completed feature records.")
    if metadata.get("selected_complete") is False or metadata.get("status") in ("building", "failed", "incomplete"):
        raise ValueError("Feature caching is incomplete. Resume cache_features.py first.")
    partial = metadata.get("status") == "partial" or metadata.get("partial", False)
    source_train_count = metadata.get("source_counts", {}).get("train")
    selected_train_count = metadata.get("selected_counts", {}).get("train")
    training_partial = (selected_train_count < source_train_count
                        if source_train_count is not None and selected_train_count is not None else partial)
    if args.epochs is not None and training_partial:
        raise ValueError("--epochs requires a complete training cache. A limited cache may only be used with --max-steps.")
    training = [row for row in metadata["records"] if row["split"] == "train"]
    validation = [row for row in metadata["records"] if row["split"] == "validation"][:args.validation_limit]
    if not training or not validation:
        raise ValueError("Both cached train and validation rows are required.")
    train_groups = {row["group"] for row in training}
    train_images = {row.get("image_sha256") for row in training} - {None}
    if any(row["group"] in train_groups or row.get("image_sha256") in train_images
           for row in metadata["records"] if row["split"] in ("validation", "test")):
        raise ValueError("Held-out data overlaps training book groups or image hashes.")
    paths = {row["id"]: feature_path(cache, row) for row in training + validation}
    if len(paths) != len(training + validation):
        raise ValueError("Feature record IDs must be unique.")
    output = args.output.resolve()
    if args.resume:
        args.resume = args.resume.resolve()
        if not output.is_dir() or args.resume.parent != output:
            raise ValueError("Resume checkpoint must be a direct child of the original output directory.")
    elif output.exists():
        raise FileExistsError(f"Output exists; choose a new directory or explicit --resume: {output}")

    import torch
    import torch.nn.functional as F
    from diffusers import SanaTransformer2DModel, FlowMatchEulerDiscreteScheduler, SanaPipeline
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import load_file

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if device.type == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA with BF16 support is required; run preflight.py.")
        free, total = torch.cuda.mem_get_info()
        if free < args.min_free_gib * 2**30:
            raise RuntimeError(f"Only {free/2**30:.2f}/{total/2**30:.2f}GiB GPU memory free; required preflight {args.min_free_gib:.2f}. Close other GPU jobs yourself; no process is stopped. Lower resolution or explicitly choose/cache a smaller model if needed.")
    elif args.use_8bit_adam:
        raise ValueError("CPU implementation tests use torch AdamW, not 8-bit Adam.")
    snapshot = Path(metadata["snapshot_path"])
    if not snapshot.is_dir():
        raise FileNotFoundError("Pinned model snapshot is unavailable locally; restore the recorded model revision.")
    steps_per_epoch = math.ceil(len(training) / args.accumulation)
    total_steps = args.max_steps if args.max_steps is not None else args.epochs * steps_per_epoch
    config = {"cache_fingerprint": metadata["fingerprint"],
              "cache_manifest_path": str(cache / "cache_manifest.json"),
              "cache_manifest_sha256": digest(cache / "cache_manifest.json"),
              "model_id": metadata["model_id"], "revision": metadata["revision"],
              "snapshot_path": str(snapshot), "cache_config": metadata["config"],
              "rank": args.rank, "lora_alpha": args.rank, "lora_dropout": 0.,
              "target_modules": ["to_q", "to_k", "to_v"],
              "precision": str(dtype), "device": args.device, "batch_size": 1,
              "accumulation": args.accumulation, "learning_rate": args.learning_rate,
              "warmup_steps": args.warmup_steps, "weight_decay": args.weight_decay,
              "max_grad_norm": args.max_grad_norm, "seed": args.seed,
              "optimizer": "AdamW8bit" if args.use_8bit_adam else "AdamW",
              "gradient_checkpointing": not args.no_gradient_checkpointing,
              "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
              "epochs_requested": args.epochs, "mode": "full_epochs" if args.epochs else "step_limited_pilot",
              "training_cache_partial": training_partial,
              "training_rows": len(training), "training_ids": [row["id"] for row in training],
              "validation_ids": [row["id"] for row in validation],
              "validation_seed": args.validation_seed, "validation_every": args.validation_every,
              "checkpoint_every": args.checkpoint_every,
              "scheduler": "FlowMatchEulerDiscreteScheduler", "flow_shift": args.flow_shift,
              "flow_sampling": "uniform scheduler index", "flow_target": "noise-minus-scaled-clean-latent",
              "validation_metric": "fixed-noise held-out flow MSE; not caption accuracy",
              "training_script_sha256": digest(Path(__file__))}
    if not args.resume:
        output.mkdir(parents=True)
        atomic_json(output / "config.json", config)
    def log(**entry):
        entry["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with (output / "progress.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(entry, ensure_ascii=False, allow_nan=False), flush=True)
    try:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4)
        log(event="loading_transformer_only", model=metadata["model_id"], revision=metadata["revision"])
        model = SanaTransformer2DModel.from_pretrained(snapshot, subfolder="transformer",
                    torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True)
        if getattr(model.config, "guidance_embeds", False):
            raise ValueError("Guidance-distilled SANA variant needs a separately specified guidance training protocol.")
        model.requires_grad_(False)
        model.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.,
                          init_lora_weights="gaussian", target_modules=config["target_modules"]))
        model.to(device=device, dtype=dtype)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if not args.no_gradient_checkpointing:
            model.enable_gradient_checkpointing()
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not params:
            raise RuntimeError("No trainable LoRA parameters found.")
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer_type = bnb.optim.AdamW8bit
        else:
            optimizer_type = torch.optim.AdamW
        optimizer = optimizer_type(params, lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler_path = snapshot / "scheduler/scheduler_config.json"
        scheduler_source = read_json(scheduler_path) if scheduler_path.exists() else {}
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=int(scheduler_source.get("num_train_timesteps", 1000)),
                    shift=args.flow_shift, use_dynamic_shifting=False)
        log(event="model_ready", trainable_parameters=sum(p.numel() for p in params),
            training_rows=len(training), total_steps=total_steps,
            source_inference_flow_shift=scheduler_source.get("flow_shift"), training_flow_shift=args.flow_shift)

        def loss_for(row, validation_mode=False):
            cached = load_file(str(paths[row["id"]]), device="cpu")
            clean = cached["latents"].unsqueeze(0).to(device=device, dtype=dtype)
            embedding = cached["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)
            mask = cached["prompt_attention_mask"].unsqueeze(0).to(device=device)
            if clean.ndim != 4 or embedding.ndim != 3 or mask.shape != embedding.shape[:2]:
                raise ValueError(f"Invalid cached tensor shape: {row['id']}")
            if clean.shape[1] != model.config.in_channels or embedding.shape[-1] != model.config.caption_channels:
                raise ValueError("Cache tensor channels differ from pinned transformer configuration.")
            if not all(torch.isfinite(value).all() for value in (clean, embedding)):
                raise RuntimeError("Nonfinite cached tensor.")
            if validation_mode:
                generator = torch.Generator(device="cpu").manual_seed(validation_seed(row["id"], args.validation_seed))
                noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32).to(device=device, dtype=dtype)
                indices = torch.randint(len(scheduler.timesteps), (1,), generator=generator)
            else:
                noise = torch.randn_like(clean)
                indices = torch.randint(len(scheduler.timesteps), (1,))
            noisy, target, timestep = make_flow_batch(clean, scheduler, noise, indices)
            amp = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else contextlib.nullcontext()
            with amp:
                prediction = model(hidden_states=noisy, encoder_hidden_states=embedding,
                    encoder_attention_mask=mask, timestep=timestep, return_dict=False)[0]
                value = F.mse_loss(prediction.float(), target.float(), reduction="mean")
            if not torch.isfinite(value):
                raise RuntimeError("Nonfinite flow loss.")
            return value

        def evaluate():
            was_training = model.training
            model.eval()
            with torch.no_grad():
                value = sum(loss_for(row, True).item() for row in validation) / len(validation)
            model.train(was_training)
            return value

        if args.resume:
            state = checkpoint_restore(args.resume, model, optimizer, config, device)
            if sorted(state["order"]) != list(range(len(training))) or not 0 <= state["cursor"] <= len(training):
                raise ValueError("Invalid restored data order/cursor.")
            log(event="resumed", step=state["step"], epoch=state["epoch"], cursor=state["cursor"])
        else:
            model.disable_adapters()
            baseline = evaluate()
            model.enable_adapters()
            state = {"step": 0, "epoch": 0, "cursor": 0, "order": data_order(len(training), args.seed, 0),
                     "examples_seen": 0, "baseline_validation_loss": baseline,
                     "initial_adapter_sha256": adapter_digest(model), "training_elapsed_seconds": 0.}
            atomic_json(output / "before.json", {"validation_flow_mse": baseline,
                        "n": len(validation), "seed": args.validation_seed,
                        "meaning": config["validation_metric"]})
            log(event="baseline_validation", loss=baseline, n=len(validation))
        if state["step"] > total_steps:
            raise ValueError("Checkpoint exceeds the requested total steps.")
        if state["step"] == total_steps:
            log(event="recovering_finalization", step=state["step"], optimizer_steps_to_run=0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        elapsed_before = state["training_elapsed_seconds"]
        last_checkpoint_step = state["step"] if args.resume else -1
        final_loss = None
        for step in range(state["step"] + 1, total_steps + 1):
            if state["cursor"] >= len(training):
                state["epoch"] += 1
                state["cursor"] = 0
                state["order"] = data_order(len(training), args.seed, state["epoch"])
            indices = state["order"][state["cursor"]:state["cursor"] + args.accumulation]
            step_started = time.perf_counter()
            total_loss = 0.
            for index in indices:
                value = loss_for(training[index])
                total_loss += value.detach().item()
                (value / len(indices)).backward()
            gradient = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            if not torch.isfinite(gradient):
                raise RuntimeError("Nonfinite LoRA gradient.")
            warmup = min(1., step / max(1, args.warmup_steps))
            decay = max(.1, (total_steps - step + 1) / max(1, total_steps - args.warmup_steps))
            learning_rate = args.learning_rate * warmup * min(1., decay)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if not all(torch.isfinite(parameter).all() for parameter in params):
                raise RuntimeError("Optimizer produced nonfinite adapter parameters.")
            if device.type == "cuda":
                torch.cuda.synchronize()
            state.update(step=step, cursor=state["cursor"] + len(indices),
                         examples_seen=state["examples_seen"] + len(indices),
                         training_elapsed_seconds=elapsed_before + time.perf_counter() - started)
            log(event="train", step=step, total_steps=total_steps, epoch=state["epoch"],
                examples_seen=state["examples_seen"], loss=total_loss/len(indices),
                grad_norm=float(gradient), learning_rate=learning_rate,
                seconds=time.perf_counter()-step_started,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else 0.)
            stop_now = args.stop_after_steps is not None and step >= args.stop_after_steps
            if step % args.validation_every == 0 or step == total_steps or stop_now:
                final_loss = evaluate()
                log(event="validation", step=step, loss=final_loss, n=len(validation))
            if step % args.checkpoint_every == 0 or step == total_steps or stop_now:
                destination = output / f"checkpoint-{step}"
                checkpoint_save(destination, model, optimizer, state, config, device)
                last_checkpoint_step = step
                atomic_json(output / "latest_checkpoint.json", {"path": str(destination), "step": step})
            if stop_now and step < total_steps:
                atomic_json(output / "status.json", {"status": "stopped_after_checkpoint", "step": step,
                            "requested_total_steps": total_steps, "resume": str(output / f"checkpoint-{step}")})
                log(event="intentional_stop", step=step, requested_total_steps=total_steps)
                return
        final_adapter = output / "final_adapter"
        changed = adapter_digest(model) != state["initial_adapter_sha256"]
        if not changed:
            raise RuntimeError("Adapter parameters did not change; training cannot be declared successful.")
        if final_loss is None:
            final_loss = evaluate()
            log(event="finalization_validation", step=state["step"], loss=final_loss, n=len(validation))
        published = publish_final_adapter(final_adapter, model, config)
        result = {"status": "complete", "step": state["step"], "examples_seen": state["examples_seen"],
                  "training_elapsed_seconds": state["training_elapsed_seconds"],
                  "baseline_validation_flow_mse": state["baseline_validation_loss"],
                  "validation_flow_mse": final_loss, "n": len(validation), "adapter_changed": changed,
                  "adapter_sha256": adapter_digest(model), "final_adapter": str(final_adapter),
                  "config_sha256": config_digest(config),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else 0.,
                  "meaning": "Execution completed; flow-loss improvement is not caption/image quality validation."}
        if (output / "after.json").exists():
            previous = read_json(output / "after.json")
            for key in ("status", "step", "examples_seen", "adapter_changed", "adapter_sha256",
                        "final_adapter", "config_sha256", "baseline_validation_flow_mse", "n"):
                if previous.get(key) != result[key]:
                    raise ValueError("Existing completion report conflicts with the verified final checkpoint.")
            # Preserve measured training time/peak memory from the original completion.
            result = previous
        else:
            atomic_json(output / "after.json", result)
        if not (output / "status.json").exists() or read_json(output / "status.json") != result:
            atomic_json(output / "status.json", result)
        log(event="final_adapter_published" if published else "final_adapter_verified", step=state["step"])
        log(event="complete", **{key:value for key,value in result.items() if key != "status"})
    except BaseException as error:
        oom = isinstance(error, torch.cuda.OutOfMemoryError)
        hint = "Reduce resolution and recache, or explicitly select a smaller SANA model; do not run another GPU job simultaneously. No model is switched automatically." if oom else "Inspect progress.jsonl; original data/base weights are untouched."
        atomic_json(output / "failure.json", {"type": type(error).__name__, "message": str(error), "hint": hint})
        raise


if __name__ == "__main__":
    main()
