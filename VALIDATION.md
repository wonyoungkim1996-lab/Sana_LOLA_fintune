# Validation record — 2026-09-15

## Scope

This package trains the **SANA 1.5 4.8B teacher with BF16 base weights and LoRA**.
It does not implement 4-bit QLoRA or Student distillation. Previous 1.6B/600M
distillation smoke results are linked in [PREVIOUS_EXPERIMENTS.md](docs/PREVIOUS_EXPERIMENTS.md).

The local validation environment was Windows, Python 3.11.8, PyTorch
2.5.1+cu121, RTX 4070 SUPER with 12,282 MiB VRAM, and the package versions
pinned in `requirements.txt`. A clean installation on the company server has
not been tested. Run `preflight.py` and the smoke command there before training.

Model: `Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers`.
Revision: `9468102c3cebb657f8c4b5f1e5a71e989a15f10d`.
The VAE is loaded directly in FP32; transformer and text encoder use BF16.

## CPU checks: PASS

```bash
python -m unittest discover -s . -p "test_*.py" -v
```

45 tests passed in 39.886 seconds. Coverage includes exact image/JSON basename
pairing, ISBN grouping, split leakage rejection, source preservation, invalid
caption rejection, image decoding and letterboxing, cache integrity and journal
recovery, inference settings, and the actual installed SANA input validator.

A separate **11,092-parameter randomly initialized CPU SANA** verifies two-step
continuous/resumed adapter tensor equality and Diffusers adapter-reload forward
equality. It also verifies recovery after the last checkpoint but before final
publication, unchanged final files on an identical completed resume, and rejection
of conflicting final config/tensors. These CPU equality checks are not 4.8B GPU
quality measurements.

## Data fixture

Ten real source images were copied into a separate execution fixture and paired
by exact basename with ten English captions from the earlier caption comparison.
These English texts are execution fixtures, not a certified translation gold set.
All 10 pairs passed preparation and were retained: 5 train and 5 validation book
groups, no test rows. The smoke caches only 4 train and 2 validation records.
At accumulation 1 and two optimizer steps, only **2 training examples are used**.

This does not validate the complete 40,001-pair English corpus. Translation of
that full corpus and its complete pairing audit must finish before full training.

## Actual 4.8B GPU execution

**PASS**: cache → one optimizer update → checkpoint → fresh-process resume →
second update → final adapter → paired base/LoRA images.

```bash
python run_smoke.py --data-dir runs/english_pair_fixture/data --output runs/teacher_pipeline_smoke_final_20260915 --resolution 512 --local-files-only
```

`runs/english_pair_fixture/data` above is the local 10-pair execution fixture,
which is excluded from the repository. Other machines should prepare their own
pairs and pass that output directory as `--data-dir`.

| Measurement | Observed result |
| --- | --- |
| Model / image size | Actual 4.8B teacher / 512 × 512 |
| Trainable parameters | 12,902,400 LoRA parameters |
| Optimizer | AdamW, rank 8, accumulation 1, BF16 base + FP32 adapters |
| Training updates | 2, with a process restart after update 1 |
| Stage exit codes | 0 / 0 / 0 / 0 |
| Total wall time, model already downloaded | 68.61 seconds |
| Maximum training allocation reported by PyTorch | 9.2954 GiB |
| Comparison | One held-out caption, seed 17, CFG 5, 18 sampling steps |
| Generated comparison images | 2 valid PNGs; base and LoRA hashes differ |
| Base / LoRA generation time | 10.469 / 7.015 seconds |
| Maximum comparison allocation | 9.0739 GiB |
| Adapter changed | Yes |

These are PyTorch allocated-memory peaks, not total system/driver VRAM usage.
Model loading, caching, checks and generation are included in the 68.61-second
wall time. It is not an estimate of 40,001-pair training time.

Fixed-noise validation flow MSE on two held-out records was **0.92139465 before**
and **0.92264950 after** the two updates. It increased slightly; this smoke does
**not** demonstrate learning-quality improvement. This loss is not caption
accuracy. Visual inspection of the comparison also did not establish correct
separation of the physician and king described by the caption.

The final adapter tensor digest was
`b43294bad389c64830093cc640b8139d21e901977e3a89ec05ed020d43d6b3db`.
The machine-readable local result is
`runs/teacher_pipeline_smoke_final_20260915/validation.json`; per-stage logs and
the paired gallery remain in that run directory. They are excluded from publication.

The separate `generate.py` entry point also passed with the saved final adapter
and both new example prompts in `examples/prompts.jsonl`, producing two additional
512px PNGs at 18 sampling steps (22.843 and 7.625 seconds). This exercises live
tokenization/text encoding and adapter loading without a paired input image.
Neither those two prompts nor these two outputs constitute an image-quality
evaluation set. A source-only machine-readable summary is included in
[`docs/teacher_lora_smoke_validation.json`](docs/teacher_lora_smoke_validation.json).

## Failures found during development

- The first comparison call inherited Diffusers' `negative_prompt=""` while
  providing negative embeddings. The final code explicitly passes
  `negative_prompt=None`; a regression test reproduces the old error and checks
  the corrected call against the installed pipeline validator.
- One repeat training process exited with Windows code **3221226505** after
  model loading, without a Python traceback, while CPU tests were also running.
  The cause is unresolved. The subsequent GPU run was isolated from those tests.
  A passing short rerun does not establish long-run native-runtime stability.

## What these checks establish

A smoke verifies execution, finite losses/gradients, changed adapter parameters,
checkpoint persistence, process restart, and image generation. It does not prove
caption faithfulness, image quality improvement, full-dataset convergence, or
compatibility of 1024px training with this 12GB GPU. The company GPU needs its own
pilot at the intended resolution. Validation/test books must remain outside
optimizer updates.

No original images, fixture images, English corpus, model weights, trained
adapters, local logs, credentials, or personal paths are included in the source
archive or GitHub source. Generated local artifacts remain available locally.
