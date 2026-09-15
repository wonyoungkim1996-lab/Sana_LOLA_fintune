"""Exercise 4.8B teacher LoRA: feature cache, update, process restart, paired images.

Uses a small subset of already prepared train/validation data. Does not run
Student distillation. Saves exact subprocess commands and checks actual artifacts.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    cache = output / "cache"
    training = output / "train"
    compare = output / "compare"
    python = sys.executable
    cache_cmd = [python, str(ROOT / "cache_features.py"), "--data-dir", str(args.data_dir.resolve()),
                 "--cache-dir", str(cache), "--resolution", str(args.resolution),
                 "--train-limit", "4", "--validation-limit", "2"]
    if args.local_files_only:
        cache_cmd.append("--local-files-only")
    train_cmd = [python, str(ROOT / "train_lora.py"), "--cache", str(cache / "latest.json"),
                 "--output", str(training), "--max-steps", "2", "--accumulation", "1",
                 "--checkpoint-every", "1", "--validation-every", "1", "--warmup-steps", "0"]
    stages = [cache_cmd, train_cmd + ["--stop-after-steps", "1"],
              train_cmd + ["--resume", str(training / "checkpoint-1")],
              [python, str(ROOT / "sample_compare.py"), "--cache", str(cache / "latest.json"),
               "--adapter", str(training / "final_adapter"), "--output", str(compare),
               "--count", "1", "--seeds", "17", "--steps", "18"]]
    plan = {"task": "4.8B teacher LoRA execution smoke, not Student distillation",
            "commands": stages, "training_steps": 2, "full_dataset_training": False,
            "max_train_rows": 4, "max_validation_rows": 2}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2))
        return
    if output.exists():
        raise FileExistsError("Use a new smoke output; logs and adapters are preserved")
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    started = time.monotonic()
    results = []
    for index, command in enumerate(stages, 1):
        print(f"Stage {index}/{len(stages)}: {Path(command[1]).name}", flush=True)
        with (output / f"stage_{index}.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        results.append({"stage": index, "exit_code": process.returncode})
        if process.returncode:
            (output / "validation.json").write_text(json.dumps({"passed": False, "stages": results,
                "failed_log": f"stage_{index}.log", "completed_full_training": False}, indent=2), encoding="utf-8")
            raise SystemExit(process.returncode)
    status = json.loads((training / "status.json").read_text(encoding="utf-8"))
    images = json.loads((compare / "completed.json").read_text(encoding="utf-8"))
    from PIL import Image
    generated = list(compare.glob("*/*.png"))
    for path in generated:
        with Image.open(path) as image:
            image.verify()
    passed = status["status"] == "complete" and status["adapter_changed"] and status["step"] == 2 and len(generated) == images["images"] == 2
    result = {"passed": bool(passed), "stages": results, "optimizer_steps": status["step"],
              "adapter_changed": status["adapter_changed"], "generated_images": len(generated),
              "fresh_process_resume": True, "elapsed_seconds": round(time.monotonic()-started, 2),
              "full_dataset_training": False, "image_quality_improvement_proven": False}
    (output / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
