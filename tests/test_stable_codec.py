import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from anytrain.codec.stable_codec import (
    DEFAULT_PRETRAINED_MODEL,
    StableCodec,
)
from torch import nn


class StableCodecTest(unittest.TestCase):
    def test_from_pretrained_loads_default_model(self):
        with patch(
            "anytrain.codec.stable_codec.codec._load_stable_codec_model",
            return_value=FakeStableCodecBackend,
        ):
            codec = StableCodec.from_pretrained(device="cpu")

        self.assertIsInstance(codec.model, FakeStableCodecBackend)
        self.assertEqual(FakeStableCodecBackend.kwargs["pretrained_model"], DEFAULT_PRETRAINED_MODEL)
        self.assertEqual(FakeStableCodecBackend.kwargs["device"], torch.device("cpu"))
        self.assertEqual(codec.device, torch.device("cpu"))
        self.assertEqual(codec.sample_rate, 16000)

    def test_from_pretrained_sets_posthoc_bottleneck(self):
        with patch(
            "anytrain.codec.stable_codec.codec._load_stable_codec_model",
            return_value=FakeStableCodecBackend,
        ):
            codec = StableCodec.from_pretrained(
                device="cpu",
                posthoc_bottleneck="2x15625_700bps",
            )

        self.assertTrue(codec.posthoc_bottleneck)
        self.assertEqual(codec.model.posthoc_stages, "2x15625_700bps")

    def test_from_config_loads_local_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            checkpoint = root / "model.ckpt"

            with patch(
                "anytrain.codec.stable_codec.codec._load_stable_codec_model",
                return_value=FakeStableCodecBackend,
            ):
                codec = StableCodec.from_config(
                    config,
                    ckpt_path=checkpoint,
                    device="cpu",
                )

        self.assertIsInstance(codec.model, FakeStableCodecBackend)
        self.assertEqual(FakeStableCodecBackend.kwargs["model_config_path"], str(config))
        self.assertEqual(FakeStableCodecBackend.kwargs["ckpt_path"], str(checkpoint))

    def test_encode_returns_tokens(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"))
        audio = torch.zeros((2, 1, 16))

        tokens = codec.encode(audio, normalize=False)

        self.assertEqual(tuple(tokens.shape), (2, 5, 1))
        self.assertFalse(model.normalize)
        self.assertFalse(model.posthoc)

    def test_encode_latents_exposes_upstream_latent_boundary(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"))
        audio = torch.zeros((2, 1, 16))

        latents, tokens = codec.encode_latents(audio)

        self.assertEqual(tuple(latents.shape), (2, 4, 5))
        self.assertEqual(tuple(tokens.shape), (2, 5, 1))

    def test_decode_uses_tokens(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"))
        tokens = torch.ones((2, 5, 1), dtype=torch.long)

        audio = codec.decode(tokens, posthoc_bottleneck=True)

        self.assertEqual(tuple(audio.shape), (2, 1, 16))
        self.assertTrue(model.posthoc)

    def test_reconstruct_roundtrips_tokens(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"))
        audio = torch.zeros((2, 1, 16))

        reconstructed = codec.reconstruct(audio)

        self.assertEqual(tuple(reconstructed.shape), (2, 1, 16))
        self.assertTrue(model.decode_called)

    def test_encode_rejects_non_mono_audio(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "mono"):
            codec.encode(torch.zeros((2, 2, 16)))

    def test_encode_rejects_missing_channel_axis(self):
        model = FakeStableCodecBackend(pretrained_model="fake", device=torch.device("cpu"))
        codec = StableCodec(model=model, device=torch.device("cpu"))

        with self.assertRaisesRegex(ValueError, "shape"):
            codec.encode(torch.zeros((2, 16)))


class FakeStableCodecBackend(nn.Module):
    kwargs: dict[str, object] = {}

    def __init__(self, **kwargs) -> None:
        super().__init__()
        type(self).kwargs = kwargs
        self.sample_rate = 16000
        self.posthoc_stages = None
        self.posthoc = False
        self.normalize = True
        self.decode_called = False

    def set_posthoc_bottleneck(self, stages):
        self.posthoc_stages = stages

    def encode(self, audio: torch.Tensor, *, posthoc_bottleneck: bool, normalize: bool, **kwargs):
        self.posthoc = posthoc_bottleneck
        self.normalize = normalize
        batch = audio.size(0)
        return torch.ones((batch, 4, 5)), torch.ones((batch, 5, 1), dtype=torch.long)

    def decode(self, tokens: torch.Tensor, *, posthoc_bottleneck: bool, **kwargs):
        self.decode_called = True
        self.posthoc = posthoc_bottleneck
        return torch.ones((tokens.size(0), 1, 16))


if __name__ == "__main__":
    unittest.main()
