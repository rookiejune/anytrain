from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anytrain.codec.dac.assets as dac_assets
from anytrain.codec.dac import ensure_dac_assets


class DACAssetsTest(unittest.TestCase):
    def test_ensure_assets_downloads_to_explicit_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_download(url: str, target: Path) -> None:
                target.write_text(url)

            with patch.object(dac_assets, "_download", side_effect=fake_download):
                assets = ensure_dac_assets(cache_dir=root, model_type="24khz")

            self.assertEqual(assets["cache_dir"], root)
            self.assertEqual(assets["tag"], "0.0.4")
            self.assertEqual(
                assets["checkpoint"],
                root / "weights_24khz_8kbps_0.0.4.pth",
            )
            self.assertIn("weights_24khz.pth", assets["checkpoint"].read_text())

    def test_ensure_assets_reuses_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "weights_44khz_8kbps_0.0.1.pth"
            checkpoint.write_text("existing")

            with patch.object(
                dac_assets,
                "_download",
                side_effect=AssertionError("unexpected download"),
            ):
                assets = ensure_dac_assets(cache_dir=root)

            self.assertEqual(assets["checkpoint"], checkpoint)
            self.assertEqual(checkpoint.read_text(), "existing")

    def test_local_files_only_exposes_missing_checkpoint(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(FileNotFoundError, "not available locally"),
        ):
            ensure_dac_assets(cache_dir=tmp, local_files_only=True)

    def test_unknown_preset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no latest checkpoint"):
            ensure_dac_assets(model_type="24khz", model_bitrate="16kbps")


if __name__ == "__main__":
    unittest.main()
