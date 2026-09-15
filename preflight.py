"""Read-only environment report. Does not load model weights or start training."""
import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    required = ["torch", "diffusers", "transformers", "peft", "accelerate", "safetensors", "Pillow", "huggingface-hub", "sentencepiece", "numpy"]
    packages, missing = {}, []
    for name in required + ["bitsandbytes", "protobuf"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            if name in required:
                missing.append(name)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        gpu_report = gpu.stdout.strip() or gpu.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        gpu_report = str(exc)
    free = shutil.disk_usage(Path(__file__).resolve().parent).free / 2**30
    report = {"python": sys.version, "executable": sys.executable, "packages": packages,
              "missing_required": missing, "gpu_snapshot": gpu_report, "free_disk_gib": round(free, 2),
              "large_model_execution_tested": False,
              "notes": ["4.8B BF16 weights alone are about 8.94 GiB; activations and runtime need additional memory.",
                        "12 GiB compatibility must be measured; no automatic model downgrade is performed.",
                        "Cache size depends on caption token counts; padding rows can be omitted from storage.",
                        "protobuf is optional for the existing fast-tokenizer JSON, and may be needed for tokenizer conversion.",
                        "Install this project in its own virtual environment. Do not upgrade an active training environment."]}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return bool(missing)


if __name__ == "__main__":
    raise SystemExit(main())
