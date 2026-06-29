import tempfile
import unittest

import torch

from anytrain.tokenizer.codec_bpe import CodecBPE

try:
    import tokenizers
except ImportError:
    tokenizers = None


def corpus():
    return [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class CodecBPERegressionTest(unittest.TestCase):
    def test_from_pretrained_accepts_state_file(self):
        bpe = CodecBPE.train(corpus(), codebook_sizes=(16,), vocab_size=5)

        with tempfile.TemporaryDirectory() as tmp:
            out = bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(out / "codec_bpe.json")

        self.assertEqual(loaded.to_dict(), bpe.to_dict())

    def test_progress_keeps_same_result(self):
        plain = CodecBPE.train(corpus(), codebook_sizes=(16,), vocab_size=5, show_progress=False)
        with_progress = CodecBPE.train(
            corpus(),
            codebook_sizes=(16,),
            vocab_size=5,
            show_progress=True,
        )

        self.assertEqual(with_progress.to_dict(), plain.to_dict())

    def test_repeat_interleave_uses_expansion_counts(self):
        bpe = CodecBPE.train(corpus(), codebook_sizes=(16,), vocab_size=5)
        x = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        token_ids = torch.tensor([3, 4])

        expanded_x, frames = bpe.repeat_interleave(x, token_ids, dim=0)

        self.assertTrue(
            torch.equal(
                expanded_x,
                torch.tensor(
                    [
                        [10.0, 11.0],
                        [10.0, 11.0],
                        [20.0, 21.0],
                        [20.0, 21.0],
                        [20.0, 21.0],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(frames, torch.tensor([[1], [2], [1], [2], [3]]))
        )


if __name__ == "__main__":
    unittest.main()
