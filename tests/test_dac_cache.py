import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.codec.dac._cache import (
    ANYTRAIN_HOME_ENV,
    DEFAULT_DAC_HOME,
    cache_dir,
)


class DACCacheTest(unittest.TestCase):
    def test_explicit_cache_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            path = cache_dir(root / "explicit")

            self.assertEqual(path, root / "explicit")

    def test_anytrain_home_controls_default_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            with patch.dict(os.environ, {ANYTRAIN_HOME_ENV: str(root)}, clear=True):
                path = cache_dir()

            self.assertEqual(path, root / "dac")

    def test_default_cache_dir_uses_packaged_home(self):
        with patch.dict(os.environ, {}, clear=True):
            path = cache_dir()

        self.assertEqual(path, DEFAULT_DAC_HOME)

    def test_empty_anytrain_home_fails(self):
        with (
            patch.dict(os.environ, {ANYTRAIN_HOME_ENV: ""}, clear=True),
            self.assertRaisesRegex(ValueError, ANYTRAIN_HOME_ENV),
        ):
            cache_dir()


if __name__ == "__main__":
    unittest.main()
