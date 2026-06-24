import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.codec.longcat.assets import LongCatAssets, LongCatConfigPaths
from anytrain.codec.longcat.codec import LongCatAudioCodec


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
                return "encoder"

            def fake_load_decoder(config_path, device):
                self.assertEqual(config_path, str(configs.decoder_16k_4codebooks))
                self.assertEqual(os.environ["LONGCAT_AUDIO_CODEC_CKPT_DIR"], str(assets.ckpt_dir))
                return "decoder"

            def fake_ensure_assets(**kwargs):
                self.assertEqual(kwargs["decoders"], ("16k_4codebooks",))
                return assets

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("LONGCAT_AUDIO_CODEC_CKPT_DIR", None)
                with (
                    patch("anytrain.codec.longcat.codec.ensure_longcat_assets", side_effect=fake_ensure_assets),
                    patch(
                        "anytrain.codec.longcat.codec._load_longcat_loaders",
                        return_value=(fake_load_encoder, fake_load_decoder),
                    ),
                ):
                    codec = LongCatAudioCodec.from_pretrained(device="cpu")

                self.assertEqual(codec.encoder, "encoder")
                self.assertEqual(codec.decoders["16k_4codebooks"], "decoder")
                self.assertNotIn("LONGCAT_AUDIO_CODEC_CKPT_DIR", os.environ)


if __name__ == "__main__":
    unittest.main()
