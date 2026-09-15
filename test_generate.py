import argparse
import json
from pathlib import Path
import tempfile
import unittest

from generate import MODEL, REVISION, read_prompts, resolve_settings


class InferenceContractTests(unittest.TestCase):
    def args(self, **kwargs):
        defaults = dict(prompt=None, caption_json=None, prompts_jsonl=None, adapter=None,
                        model_id=None, revision=None, resolution=None)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_paired_caption_preserved_and_filename_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "03_01T_01S_9788959991006_57265.json"
            caption = "A rabbit holds 2 apples."
            path.write_text(json.dumps({"imageCaption": caption}), encoding="utf-8")
            self.assertEqual(read_prompts(self.args(caption_json=path)), [{"id": path.stem, "imageCaption": caption}])

    def test_duplicate_id_and_path_escape_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            for ids in [("one", "ONE"), ("../escape", "two")]:
                path.write_text("\n".join(json.dumps({"id": i, "imageCaption": "A rabbit."}) for i in ids), encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_prompts(self.args(prompts_jsonl=path))

    def test_nonenglish_and_extra_fields_refused(self):
        with self.assertRaises(ValueError):
            read_prompts(self.args(prompt="\ud1a0\ub07c\uac00 \uc788\ub2e4."))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.json"
            path.write_text('{"imageCaption":"A rabbit.","QA":"ignored?"}', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_prompts(self.args(caption_json=path))

    def test_adapter_inherits_condition_and_rejects_model_resolution_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "pytorch_lora_weights.safetensors").touch()
            config = {"model_id": MODEL, "revision": REVISION,
                      "cache_config": {"caption_language": "en", "resolution": 512,
                                       "prefix": "Illustration. ", "max_sequence_length": 300}}
            (adapter / "config.json").write_text(json.dumps(config), encoding="utf-8")
            settings = resolve_settings(self.args(adapter=adapter))
            self.assertEqual(settings["prefix"], "Illustration. ")
            self.assertEqual(settings["resolution"], 512)
            for kwargs in ({"model_id": "other/model"}, {"revision": "other"}, {"resolution": 1024}):
                with self.assertRaises(ValueError):
                    resolve_settings(self.args(adapter=adapter, **kwargs))


if __name__ == "__main__":
    unittest.main()
