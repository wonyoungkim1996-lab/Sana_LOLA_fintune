"""Real, tiny CPU-only trainer/recovery/LoRA-format integration test.

Creates a one-block randomly initialized SANA model and synthetic tensors inside
a temporary directory. It never downloads weights, loads the real SANA checkpoint,
uses an image dataset, or makes CUDA visible to the training subprocesses.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


class TinyTrainingIntegration(unittest.TestCase):
    def test_cpu_training_resume_and_diffusers_adapter_reload(self):
        import torch
        from diffusers import SanaTransformer2DModel, SanaPipeline, FlowMatchEulerDiscreteScheduler
        from peft import LoraConfig, set_peft_model_state_dict
        from safetensors.torch import load_file, save_file

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="sana_lora_cpu_integration_") as temporary:
            workspace = Path(temporary)
            snapshot = workspace / "synthetic_snapshot"
            torch.manual_seed(107)
            tiny_model = SanaTransformer2DModel(
                in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=8,
                num_layers=1, num_cross_attention_heads=2, cross_attention_head_dim=8,
                cross_attention_dim=16, caption_channels=8, mlp_ratio=2., sample_size=4,
                patch_size=1, dropout=0., guidance_embeds=False, qk_norm=None)
            parameter_count = sum(parameter.numel() for parameter in tiny_model.parameters())
            self.assertLess(parameter_count, 100_000, "Fixture must remain tiny, not a production checkpoint.")
            tiny_model.save_pretrained(snapshot / "transformer", safe_serialization=True)
            FlowMatchEulerDiscreteScheduler(num_train_timesteps=20, shift=1.).save_pretrained(snapshot / "scheduler")
            del tiny_model

            cache = workspace / "cache"
            (cache / "features").mkdir(parents=True)
            generator = torch.Generator(device="cpu").manual_seed(109)
            records = []
            for index, split in enumerate(("train", "train", "validation")):
                identity = f"synthetic_{index}"
                # Variable-length text caches exercise batch-one conditioning.
                length = 4 + index
                path = cache / "features" / f"{identity}.safetensors"
                save_file({"latents": torch.randn((4, 4, 4), generator=generator).to(torch.bfloat16),
                           "prompt_embeds": torch.randn((length, 8), generator=generator).to(torch.bfloat16),
                           "prompt_attention_mask": torch.ones(length, dtype=torch.long)}, str(path))
                records.append({"id": identity, "group": f"book_{index}", "split": split,
                                "image_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                                "caption": f"Synthetic CPU fixture {index}",
                                "feature_file": str(path.relative_to(cache)), "feature_sha256": file_hash(path)})
            fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
            write_json(cache / "cache_manifest.json", {
                "fingerprint": fingerprint, "model_id": "synthetic-cpu-sana-test",
                "revision": "random-fixture-seed107", "snapshot_path": str(snapshot),
                "config": {"resolution": 128, "max_sequence_length": 8, "prefix": "",
                           "caption_language": "synthetic", "embedding_storage": "trimmed_to_last_attended_token"},
                "status": "complete", "partial": False, "selected_complete": True,
                "source_counts": {"train": 2, "validation": 1},
                "selected_counts": {"train": 2, "validation": 1}, "records": records})
            pointer = workspace / "latest.json"
            write_json(pointer, {"cache_path": str(cache), "fingerprint": fingerprint})

            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                               HF_DATASETS_OFFLINE="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
            common = [sys.executable, str(ROOT / "train_lora.py"), "--cache", str(pointer),
                      "--device", "cpu", "--max-steps", "2", "--accumulation", "1", "--rank", "2",
                      "--warmup-steps", "0", "--validation-limit", "1", "--validation-every", "1",
                      "--checkpoint-every", "1"]

            def run(output, *extra):
                result = subprocess.run(common + ["--output", str(output)] + list(extra),
                                        env=environment, cwd=ROOT, capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=120)
                self.assertEqual(result.returncode, 0,
                                 f"Tiny trainer failed:\n{result.stdout[-6000:]}\n{result.stderr[-6000:]}")
                return result

            continuous = workspace / "continuous"
            recovered = workspace / "recovered"
            run(continuous)
            run(recovered, "--stop-after-steps", "1")
            stopped_status = json.loads((recovered / "status.json").read_text())
            self.assertEqual(stopped_status["status"], "stopped_after_checkpoint")
            self.assertEqual(stopped_status["step"], 1)
            self.assertFalse((recovered / "final_adapter").exists())
            stopped = torch.load(recovered / "checkpoint-1/training_state.pt", map_location="cpu", weights_only=False)
            self.assertEqual((stopped["step"], stopped["cursor"], stopped["epoch"]), (1, 1, 0))
            self.assertEqual(sorted(stopped["order"]), [0, 1])
            self.assertTrue(stopped["optimizer"]["state"])
            self.assertIsNotNone(stopped["rng"]["torch"])
            self.assertIsNone(stopped["rng"]["cuda"])
            run(recovered, "--resume", str(recovered / "checkpoint-1"))
            before = json.loads((continuous / "before.json").read_text())
            after = json.loads((continuous / "after.json").read_text())
            resumed_after = json.loads((recovered / "after.json").read_text())
            self.assertTrue(after["adapter_changed"])
            self.assertTrue(resumed_after["adapter_changed"])
            self.assertEqual(after["step"], 2)
            self.assertEqual(after["examples_seen"], 2)
            self.assertEqual(after["peak_allocated_gib"], 0.)
            self.assertTrue(torch.isfinite(torch.tensor(before["validation_flow_mse"])))
            self.assertTrue(torch.isfinite(torch.tensor(after["validation_flow_mse"])))
            self.assertEqual(after["validation_flow_mse"], resumed_after["validation_flow_mse"])

            complete_weights = load_file(str(continuous / "checkpoint-2/adapter_training.safetensors"))
            recovered_weights = load_file(str(recovered / "checkpoint-2/adapter_training.safetensors"))
            self.assertEqual(set(complete_weights), set(recovered_weights))
            for name in complete_weights:
                self.assertTrue(torch.equal(complete_weights[name], recovered_weights[name]), name)
            final_weights = load_file(str(continuous / "final_adapter/pytorch_lora_weights.safetensors"))
            self.assertTrue(final_weights)
            self.assertTrue(all(name.startswith("transformer.") for name in final_weights))

            # Simulate interruption after the final optimizer checkpoint but before
            # final adapter/report publication. Every changed path is fixture-owned.
            last_checkpoint = continuous / "checkpoint-2"
            checkpoint_hashes = {path.name: file_hash(path) for path in last_checkpoint.iterdir() if path.is_file()}
            for name in ("final_adapter", "after.json", "status.json"):
                source = (continuous / name).resolve()
                backup = (workspace / ("pre_interruption_" + name)).resolve()
                self.assertTrue(source.is_relative_to(workspace.resolve()))
                self.assertTrue(backup.is_relative_to(workspace.resolve()))
                source.rename(backup)
            staging = continuous / "final_adapter.tmp-interrupted-fixture"
            staging.mkdir()
            (staging / "partial.txt").write_text("Interrupted private staging data", encoding="utf-8")
            run(continuous, "--resume", str(last_checkpoint))
            finalized_after = json.loads((continuous / "after.json").read_text())
            self.assertEqual(finalized_after["step"], 2)
            self.assertEqual(finalized_after["examples_seen"], 2)
            self.assertEqual(finalized_after["adapter_sha256"], after["adapter_sha256"])
            self.assertEqual(finalized_after["validation_flow_mse"], after["validation_flow_mse"])
            self.assertEqual(checkpoint_hashes,
                             {path.name: file_hash(path) for path in last_checkpoint.iterdir() if path.is_file()})
            published = continuous / "final_adapter"
            self.assertTrue((published / "publication.json").is_file())
            recovered_final = load_file(str(published / "pytorch_lora_weights.safetensors"))
            for name in final_weights:
                self.assertTrue(torch.equal(final_weights[name], recovered_final[name]), name)
            self.assertEqual((staging / "partial.txt").read_text(), "Interrupted private staging data")
            publication_hashes = {path.name: file_hash(path) for path in published.iterdir() if path.is_file()}
            report_hashes = {name: file_hash(continuous / name) for name in ("after.json", "status.json")}
            run(continuous, "--resume", str(last_checkpoint))
            self.assertEqual(publication_hashes,
                             {path.name: file_hash(path) for path in published.iterdir() if path.is_file()})
            self.assertEqual(report_hashes, {name: file_hash(continuous / name) for name in report_hashes})
            events = [json.loads(line) for line in (continuous / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(sum(event["event"] == "train" for event in events), 2,
                             "Finalization recovery must not perform additional optimizer steps.")
            self.assertEqual(sum(event["event"] == "recovering_finalization" for event in events), 2)

            # Verify the saved public Diffusers format produces the same forward
            # result as loading our internal PEFT training-state representation.
            reference_model = SanaTransformer2DModel.from_pretrained(snapshot, subfolder="transformer",
                                                                     local_files_only=True)
            reference_model.add_adapter(LoraConfig(r=2, lora_alpha=2, lora_dropout=0.,
                init_lora_weights="gaussian", target_modules=["to_q", "to_k", "to_v"]))
            loaded = set_peft_model_state_dict(reference_model, complete_weights, adapter_name="default")
            self.assertFalse(loaded.unexpected_keys)
            reference_model.eval()
            from train_lora import publish_final_adapter
            config = json.loads((continuous / "config.json").read_text())
            config_path = published / "config.json"
            original_config_bytes = config_path.read_bytes()
            write_json(config_path, dict(config, rank=999))
            conflicting_config_bytes = config_path.read_bytes()
            with self.assertRaisesRegex(ValueError, "configuration conflicts"):
                publish_final_adapter(published, reference_model, config)
            self.assertEqual(config_path.read_bytes(), conflicting_config_bytes)
            config_path.write_bytes(original_config_bytes)
            weights_path = published / "pytorch_lora_weights.safetensors"
            original_weight_bytes = weights_path.read_bytes()
            damaged = {name: value.clone() for name, value in final_weights.items()}
            damaged[next(iter(damaged))].reshape(-1)[0] += 1.
            save_file(damaged, str(weights_path))
            conflicting_weight_hash = file_hash(weights_path)
            with self.assertRaisesRegex(ValueError, "tensors conflict"):
                publish_final_adapter(published, reference_model, config)
            self.assertEqual(file_hash(weights_path), conflicting_weight_hash)
            weights_path.write_bytes(original_weight_bytes)
            public_model = SanaTransformer2DModel.from_pretrained(snapshot, subfolder="transformer",
                                                                  local_files_only=True)
            pipeline = SanaPipeline(tokenizer=None, text_encoder=None, vae=None,
                                    transformer=public_model,
                                    scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=20))
            pipeline.load_lora_weights(continuous / "final_adapter", adapter_name="loaded_test",
                                       local_files_only=True)
            pipeline.transformer.eval()
            inputs = {"hidden_states": torch.randn((1, 4, 4, 4), generator=generator),
                      "encoder_hidden_states": torch.randn((1, 5, 8), generator=generator),
                      "encoder_attention_mask": torch.ones((1, 5), dtype=torch.long),
                      "timestep": torch.tensor([7.], dtype=torch.float32), "return_dict": False}
            with torch.no_grad():
                expected = reference_model(**inputs)[0]
                actual = pipeline.transformer(**inputs)[0]
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
            print(json.dumps({"fixture": "tiny_random_cpu_sana", "parameters": parameter_count,
                              "continuous_and_resumed_adapters_exact": True,
                              "diffusers_lora_forward_exact": True,
                              "final_checkpoint_publication_recovered": True,
                              "completed_resume_files_unchanged": True,
                              "conflicting_final_config_and_tensors_rejected": True,
                              "steps_per_run": 2, "production_model_loaded": False,
                              "cuda_used": False}), flush=True)


if __name__ == "__main__":
    unittest.main()
