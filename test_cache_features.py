"""CPU-only feature-cache integrity checks; no pretrained models or downloads."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from cache_features import prepare_pixels, read_journal, trim_embeddings, valid_text_cache


class PixelPreparationTests(unittest.TestCase):
    def test_letterbox_preserves_both_image_edges_and_original_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'wide.png'
            image = Image.new('RGB', (8, 4), (0, 255, 0))
            for y in range(4):
                image.putpixel((0, y), (255, 0, 0))
                image.putpixel((7, y), (0, 0, 255))
            image.save(path)
            original = path.read_bytes()
            pixels = prepare_pixels(path, 8)
            self.assertEqual(pixels.shape, (1, 3, 8, 8))
            self.assertEqual(pixels.dtype, torch.float32)
            self.assertTrue(torch.equal(pixels[:, :, :2, :], torch.ones(1, 3, 2, 8)))
            self.assertTrue(torch.equal(pixels[:, :, 6:, :], torch.ones(1, 3, 2, 8)))
            self.assertEqual(pixels[0, :, 2, 0].tolist(), [1., -1., -1.])
            self.assertEqual(pixels[0, :, 5, 7].tolist(), [-1., -1., 1.])
            self.assertEqual(pixels[0, :, 3, 3].tolist(), [-1., 1., -1.])
            self.assertEqual(path.read_bytes(), original)

    def test_portrait_image_gets_white_side_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'portrait.png'
            Image.new('RGB', (4, 8), (0, 0, 0)).save(path)
            pixels = prepare_pixels(path, 8)
            self.assertTrue((pixels[:, :, :, :2] == 1).all())
            self.assertTrue((pixels[:, :, :, 6:] == 1).all())
            self.assertTrue((pixels[:, :, :, 2:6] == -1).all())


class TextCacheTests(unittest.TestCase):
    def test_trimming_keeps_internal_mask_gap_and_real_embeddings(self):
        embeddings = torch.arange(20, dtype=torch.float32).reshape(5, 4).requires_grad_()
        original = embeddings.detach().clone()
        mask = torch.tensor([1, 0, 1, 0, 0])
        result = trim_embeddings(embeddings, mask)
        self.assertEqual(result['prompt_attention_mask'].tolist(), [1, 0, 1])
        self.assertTrue(torch.equal(result['prompt_embeds'], original[:3]))
        self.assertFalse(result['prompt_embeds'].requires_grad)
        self.assertTrue(result['prompt_embeds'].is_contiguous())
        self.assertTrue(torch.equal(embeddings.detach(), original))
        self.assertEqual(mask.tolist(), [1, 0, 1, 0, 0])

    def test_valid_tensor_cache_is_accepted_only_for_its_fingerprint_and_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.safetensors'
            values = {'prompt_embeds': torch.ones(3, 4),
                      'prompt_attention_mask': torch.tensor([1, 0, 1])}
            save_file(values, str(path), metadata={'fingerprint': 'frozen', 'id': 'caption-1'})
            self.assertTrue(valid_text_cache(path, 'frozen', 'caption-1'))
            self.assertFalse(valid_text_cache(path, 'different', 'caption-1'))
            self.assertFalse(valid_text_cache(path, 'frozen', 'caption-2'))

    def test_invalid_and_nonfinite_tensor_caches_are_rejected(self):
        examples = {
            'nan_embedding': {'prompt_embeds': torch.tensor([[float('nan'), 1.]]),
                              'prompt_attention_mask': torch.ones(1)},
            'infinite_embedding': {'prompt_embeds': torch.tensor([[float('inf'), 1.]]),
                                   'prompt_attention_mask': torch.ones(1)},
            'nonbinary_mask': {'prompt_embeds': torch.ones(2, 4),
                               'prompt_attention_mask': torch.tensor([1, 2])},
            'nan_mask': {'prompt_embeds': torch.ones(2, 4),
                        'prompt_attention_mask': torch.tensor([1., float('nan')])},
            'empty_attention': {'prompt_embeds': torch.ones(2, 4),
                                'prompt_attention_mask': torch.zeros(2)},
            'empty_sequence': {'prompt_embeds': torch.ones(0, 4),
                               'prompt_attention_mask': torch.ones(0)},
            'length_mismatch': {'prompt_embeds': torch.ones(2, 4),
                                'prompt_attention_mask': torch.ones(3)},
            'wrong_rank': {'prompt_embeds': torch.ones(1, 2, 4),
                           'prompt_attention_mask': torch.ones(2)},
            'unexpected_tensor': {'prompt_embeds': torch.ones(2, 4),
                                  'prompt_attention_mask': torch.ones(2), 'extra': torch.ones(1)},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.safetensors'
            for name, values in examples.items():
                with self.subTest(name=name):
                    save_file(values, str(path), metadata={'fingerprint': 'frozen', 'id': 'caption-1'})
                    self.assertFalse(valid_text_cache(path, 'frozen', 'caption-1'))
            path.write_bytes(b'incomplete tensor header')
            self.assertFalse(valid_text_cache(path, 'frozen', 'caption-1'))
            self.assertFalse(valid_text_cache(Path(directory) / 'missing.safetensors', 'frozen', 'caption-1'))


class JournalRecoveryTests(unittest.TestCase):
    def test_interrupted_utf8_final_character_is_recovered(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feature_journal.jsonl'
            prefix = b'{"id": "first"}\n'
            broken = prefix + '{"id":"second","caption":"여'.encode('utf-8')[:-1]
            path.write_bytes(broken)
            self.assertEqual(read_journal(path), {'first': {'id': 'first'}})
            self.assertEqual(path.with_suffix('.interrupted.jsonl').read_bytes(), broken)
            self.assertEqual(path.read_bytes(), prefix)

    def test_interrupted_final_record_keeps_valid_prefix_and_original_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feature_journal.jsonl'
            rows = [{'id': 'first', 'caption': '여우'}, {'id': 'second', 'feature_file': 'features/second.safetensors'}]
            prefix = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
            interrupted = (prefix + '{"id": "third", "caption": "미완성').encode('utf-8')
            path.write_bytes(interrupted)
            recovered = read_journal(path)
            backup = path.with_suffix('.interrupted.jsonl')
            self.assertEqual(recovered, {row['id']: row for row in rows})
            self.assertEqual(backup.read_bytes(), interrupted)
            self.assertEqual([json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()], rows)
            self.assertEqual(read_journal(path), recovered)
            self.assertEqual(backup.read_bytes(), interrupted)
            with path.open('ab') as handle:
                handle.write(b'{"id": "another-interruption"')
            self.assertEqual(read_journal(path), recovered)
            self.assertEqual(backup.read_bytes(), interrupted)

    def test_corruption_in_middle_is_not_silently_discarded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'feature_journal.jsonl'
            raw = b'{"id": "first"}\nnot-json\n{"id": "third"}\n'
            path.write_bytes(raw)
            with self.assertRaises(json.JSONDecodeError):
                read_journal(path)
            self.assertEqual(path.read_bytes(), raw)
            self.assertFalse(path.with_suffix('.interrupted.jsonl').exists())

    def test_missing_journal_is_empty_without_creating_a_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'missing.jsonl'
            self.assertEqual(read_journal(path), {})
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
