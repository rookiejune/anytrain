import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.codec.longcat.cache import (
    DEFAULT_HF_HOME,
    HF_HOME_ENV,
    resolve_longcat_cache_dir,
)


class LongCatCacheTest(unittest.TestCase):
    def test_explicit_cache_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            path = resolve_longcat_cache_dir(root / "explicit")

            self.assertEqual(path, root / "explicit")

    def test_hf_home_env_controls_default_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {HF_HOME_ENV: str(root / "hf")}

            with patch.dict(os.environ, env, clear=False):
                path = resolve_longcat_cache_dir()

            self.assertEqual(path, root / "hf" / "longcat-audio-codec")

    def test_default_cache_dir_uses_default_hf_home(self):
        with patch.dict(os.environ, {}, clear=True):
            path = resolve_longcat_cache_dir()

        self.assertEqual(path, DEFAULT_HF_HOME / "longcat-audio-codec")

    def test_empty_env_fails(self):
        with (
            patch.dict(os.environ, {HF_HOME_ENV: ""}, clear=False),
            self.assertRaisesRegex(ValueError, HF_HOME_ENV),
        ):
            resolve_longcat_cache_dir()


if __name__ == "__main__":
    unittest.main()
