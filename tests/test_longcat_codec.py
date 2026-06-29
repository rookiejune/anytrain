import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from anytrain.codec.longcat.assets import LongCatAssets, LongCatConfigPaths
from anytrain.codec.longcat.codec import LongCatAudioCodec
from torch import nn


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
                    codec = LongCatAudioCodec.from_pretrained(device="cpu")

                self.assertIsInstance(codec.encoder, nn.Identity)
                self.assertIsInstance(codec.decoders["16k_4codebooks"], nn.Identity)
                self.assertNotIn("LONGCAT_AUDIO_CODEC_CKPT_DIR", os.environ)

    def test_acoustic_codes_to_features_exposes_decoder_latents_as_time_major(self):
        decoder = FakeLongCatDecoder()
        codec = LongCatAudioCodec(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        acoustic_codes = torch.tensor([[[1, 2], [3, 4]]])

        features = codec.acoustic_codes_to_features(acoustic_codes)

        self.assertEqual(tuple(features.shape), (1, 2, 3))
        self.assertTrue(torch.equal(decoder.codes, acoustic_codes))
        self.assertTrue(torch.equal(features, decoder.features.transpose(1, 2)))

    def test_decode_features_feeds_longcat_decoder_with_channel_major_latents(self):
        decoder = FakeLongCatDecoder()
        codec = LongCatAudioCodec(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": decoder},
            device=torch.device("cpu"),
            assets=_assets(),
        )
        semantic_codes = torch.tensor([[7, 8]])
        acoustic_features = torch.arange(6, dtype=torch.float).reshape(1, 2, 3)

        audio = codec.decode_features(semantic_codes, acoustic_features)

        self.assertEqual(tuple(audio.shape), (1, 1, 6))
        self.assertTrue(torch.equal(decoder.semantic_codes, semantic_codes))
        self.assertTrue(torch.equal(decoder.acoustic, acoustic_features.transpose(1, 2)))

    def test_decode_features_requires_semantic_and_acoustic_time_alignment(self):
        codec = LongCatAudioCodec(
            encoder=nn.Identity(),
            decoders={"16k_4codebooks": FakeLongCatDecoder()},
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "align on batch and time"):
            codec.decode_features(
                torch.tensor([[7, 8]]),
                torch.zeros((1, 3, 4)),
            )


class FakeLongCatDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
