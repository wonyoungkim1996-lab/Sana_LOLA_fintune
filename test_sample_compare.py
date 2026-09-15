import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sample_compare import call_cached_pipeline, guidance_embeddings, resolve_cache, resolve_feature, select_records, validate_adapter


class ComparisonIntegrityTests(unittest.TestCase):
    def test_cached_call_satisfies_installed_sana_input_validator(self):
        """Exercise our real invocation helper and Diffusers' real input checks."""
        import inspect
        import torch
        from diffusers import SanaPipeline

        class CheckOnlyPipeline:
            _callback_tensor_inputs = SanaPipeline._callback_tensor_inputs

            def __call__(self, **kwargs):
                arguments = inspect.signature(SanaPipeline.__call__).bind_partial(self, **kwargs)
                arguments.apply_defaults()
                names = inspect.signature(SanaPipeline.check_inputs).parameters
                checked = {key: arguments.arguments[key] for key in names if key != "self"}
                return SanaPipeline.check_inputs(self, **checked)

        positive = {"prompt_embeds": torch.zeros(3, 4), "prompt_attention_mask": torch.ones(3)}
        negative = {"prompt_embeds": torch.zeros(1, 4), "prompt_attention_mask": torch.ones(1)}
        embeddings = guidance_embeddings(positive, negative, "cpu", torch.float32)
        checker = CheckOnlyPipeline()
        # The pre-fix call inherits negative_prompt="" and reproduces stage 4.
        with self.assertRaisesRegex(ValueError, "Cannot forward both `negative_prompt`"):
            checker(**embeddings, height=512, width=512)
        # No weights, model forward, GPU, or network: only the installed checker.
        call_cached_pipeline(checker, embeddings, height=512, width=512)

    def test_cfg_padding_preserves_values_and_masks_padding(self):
        import torch
        positive = {"prompt_embeds": torch.arange(12).reshape(3, 4).float(), "prompt_attention_mask": torch.ones(3)}
        negative = {"prompt_embeds": torch.ones(1, 4), "prompt_attention_mask": torch.ones(1)}
        result = guidance_embeddings(positive, negative, "cpu", torch.float32)
        self.assertTrue(torch.equal(result["prompt_embeds"][0], positive["prompt_embeds"]))
        self.assertEqual(result["negative_prompt_embeds"].shape, (1, 3, 4))
        self.assertEqual(result["negative_prompt_attention_mask"].tolist(), [[1, 0, 0]])

    def test_selection_excludes_training_and_prefers_distinct_books(self):
        rows = [{"id": str(i), "group": str(i // 2), "split": "validation"} for i in range(8)]
        rows.append({"id": "train", "group": "training", "split": "train"})
        chosen = select_records(rows, "validation", 4)
        self.assertEqual(len({r["group"] for r in chosen}), 4)
        self.assertEqual(chosen, select_records(list(reversed(rows)), "validation", 4))
        self.assertTrue(all(r["split"] == "validation" for r in chosen))

    def test_tampered_features_and_wrong_adapter_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "feature.bin"
            path.write_bytes(b"original")
            row = {"feature_file": path.name, "feature_sha256": hashlib.sha256(b"original").hexdigest()}
            self.assertEqual(resolve_feature(root, row), path)
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                resolve_feature(root, row)
            (root / "config.json").write_text(json.dumps({"cache_fingerprint": "wrong"}))
            with self.assertRaises(ValueError):
                validate_adapter(root, {"fingerprint": "expected"})

    def test_relative_cache_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "abc").mkdir()
            (root / "abc/cache_manifest.json").write_text(json.dumps({"fingerprint": "abc"}))
            (root / "latest.json").write_text(json.dumps({"cache_path": "abc"}))
            path, data = resolve_cache(root / "latest.json")
            self.assertEqual(path, root / "abc")
            self.assertEqual(data["fingerprint"], "abc")

    def test_new_test_cache_requires_same_model_and_disjoint_books(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trained = {"fingerprint": "train-cache", "model_id": "tiny", "revision": "pinned",
                       "config": {"resolution": 512},
                       "records": [{"id": "a", "group": "book-a", "split": "train", "image_sha256": "image-a"}]}
            manifest = root / "cache_manifest.json"
            manifest.write_text(json.dumps(trained), encoding="utf-8")
            config = {"cache_fingerprint": "train-cache", "cache_manifest_path": str(manifest),
                      "cache_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            evaluation = {**trained, "fingerprint": "test-cache",
                          "records": [{"id": "b", "group": "book-b", "split": "test", "image_sha256": "image-b"}]}
            self.assertEqual(validate_adapter(root, evaluation), root / "config.json")
            evaluation["records"][0]["group"] = "book-a"
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_adapter(root, evaluation)
            evaluation["records"][0]["group"] = "book-b"
            evaluation["revision"] = "different"
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_adapter(root, evaluation)
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate_adapter(root, evaluation)


if __name__ == "__main__":
    unittest.main()
