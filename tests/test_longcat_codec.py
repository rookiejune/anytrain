from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from anytrain.codec.longcat.assets import LongCatAssets, LongCatConfigPaths
from anytrain.codec.longcat.codec import LongCat


class LongCatCodecTest(unittest.TestCase):
    def test_from_pretrained_sets_longcat_checkpoint_env_while_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = LongCatConfigPaths(
                encoder=root / "encoder.yaml",
                decoder_16k_4codebooks=root / "decoder16.yaml",
                decoder_24k_2codebooks=root / "decoder24_2.yaml",
                decoder_24k_4codebooks=root / "decoder24_4.yaml",
            )
            assets = LongCatAssets(
                cache_dir=root,
                ckpt_dir=root / "ckpts",
                configs=configs,
                checkpoints={},
            )

            def fake_load_encoder(config_path, device):
                self.assertEqual(config_path, str(configs.encoder))
                self.assertEqual(os.environ["LONGCAT_AUDIO_CODEC_CKPT_DIR"], str(assets.ckpt_dir))
                return nn.Identity()

            def fake_load_decoder(config_path, device):
                self.assertEqual(config_path, str(configs.decoder_16k_4codebooks))
                self.assertEqual(os.environ["LONGCAT_AUDIO_CODEC_CKPT_DIR"], str(assets.ckpt_dir))
                return nn.Identity()

            def fake_ensure_assets(**kwargs):
                self.assertEqual(kwargs["decoders"], ("16k_4codebooks",))
                return assets

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LONGCAT_AUDIO_CODEC_CKPT_DIR", None)
                with (
                    patch(
                        "anytrain.codec.longcat.codec.ensure_longcat_assets",
                        side_effect=fake_ensure_assets,
                    ),
                    patch(
                        "anytrain.codec.longcat.codec._load_longcat_loaders",
                        return_value=(fake_load_encoder, fake_load_decoder),
                    ),
                ):
                    codec = LongCat.from_pretrained(device="cpu")

                self.assertIsInstance(codec.encoder, nn.Identity)
                self.assertIsInstance(codec.decoders["16k_4codebooks"], nn.Identity)
                self.assertEqual(codec.sample_rate, 16000)
                self.assertEqual(codec.codebook_sizes, (8192, 8100, 8100, 8100))
                self.assertNotIn("LONGCAT_AUDIO_CODEC_CKPT_DIR", os.environ)

    def test_encode_returns_time_major_aligned_codebooks(self):
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        codes = codec.encode(torch.zeros((2, 1, 12)), 16000)

        self.assertEqual(tuple(codes.shape), (2, 2, 4))
        self.assertTrue(torch.equal(codes[..., 0], torch.tensor([[7, 8], [7, 8]])))
        self.assertTrue(torch.equal(codes[..., 1:], torch.ones((2, 2, 3), dtype=torch.long)))

    def test_decode_splits_time_major_codebooks_for_backend(self):
        decoder = FakeLongCatDecoder()
        codec = LongCat(
            encoder=FakeLongCatEncoder(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        codes = torch.arange(16).reshape(2, 2, 4)

        audio = codec.decode(codes)

        self.assertEqual(tuple(audio.shape), (2, 1, 6))
        self.assertTrue(torch.equal(decoder.semantic_codes, codes[..., 0]))
        self.assertTrue(torch.equal(decoder.acoustic, codes[..., 1:].transpose(1, 2)))

    def test_acoustic_codes_to_features_exposes_decoder_latents_as_time_major(self):
        decoder = FakeLongCatDecoder()
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        acoustic_codes = torch.tensor([[[1, 2, 3], [8099, 8006, 7729]]])

        features = codec.acoustic_codes_to_features(acoustic_codes)

        self.assertEqual(tuple(features.shape), (1, 2, 3))
        self.assertTrue(torch.equal(decoder.codes, acoustic_codes.transpose(1, 2)))
        self.assertTrue(torch.equal(features, decoder.features.transpose(1, 2)))

    def test_acoustic_codes_to_features_requires_time_major_full_codebooks(self):
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "3 aligned codebooks"):
            codec.acoustic_codes_to_features(torch.ones((1, 3, 2), dtype=torch.long))

    def test_decode_features_feeds_longcat_decoder_with_channel_major_latents(self):
        decoder = FakeLongCatDecoder()
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        semantic_codes = torch.tensor([[[7], [8]]])
        acoustic_features = torch.arange(6, dtype=torch.float).reshape(1, 2, 3)

        audio = codec.decode_features(semantic_codes, acoustic_features)

        self.assertEqual(tuple(audio.shape), (1, 1, 6))
        self.assertTrue(torch.equal(decoder.semantic_codes, semantic_codes[..., 0]))
        self.assertTrue(torch.equal(decoder.acoustic, acoustic_features.transpose(1, 2)))

    def test_decode_features_requires_semantic_and_acoustic_time_alignment(self):
        codec = LongCat(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "align on batch and time"):
            codec.decode_features(
                torch.tensor([[[7], [8]]]),
                torch.zeros((1, 3, 4)),
            )


class FakeLongCatDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.n_codebooks = 3
        self.acoustic_codebook_size = 90
        self.codes: torch.Tensor | None = None
        self.features: torch.Tensor | None = None
        self.semantic_codes: torch.Tensor | None = None
        self.acoustic: torch.Tensor | None = None

    def acoustic_codes_to_latents(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        self.codes = acoustic_codes
        self.features = torch.arange(6, dtype=torch.float).reshape(1, 3, 2)
        return self.features

    def forward(self, semantic_codes: torch.Tensor, acoustic: torch.Tensor) -> torch.Tensor:
        self.semantic_codes = semantic_codes
        self.acoustic = acoustic
        return torch.ones((semantic_codes.size(0), 1, acoustic.size(-1) * 3))


class FakeLongCatEncoder(nn.Module):
    def forward(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        *,
        n_acoustic_codebooks: int,
    ):
        batch = audio.size(0)
        semantic = torch.tensor([[7, 8]]).expand(batch, -1)
        acoustic = torch.ones((batch, n_acoustic_codebooks, 2), dtype=torch.long)
        return semantic, acoustic


def _assets() -> LongCatAssets:
    return LongCatAssets(
        cache_dir=Path(),
        ckpt_dir=Path(),
        configs=LongCatConfigPaths(
            encoder=Path(),
            decoder_16k_4codebooks=Path(),
            decoder_24k_2codebooks=Path(),
            decoder_24k_4codebooks=Path(),
        ),
        checkpoints={},
    )


if __name__ == "__main__":
    unittest.main()
