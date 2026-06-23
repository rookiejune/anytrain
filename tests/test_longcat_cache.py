import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from anytrain.codec.longcat.cache import (
    ANYTRAIN_CACHE_ENV,
    ANYTRAIN_LONGCAT_CACHE_ENV,
    resolve_longcat_cache_dir,
)


class LongCatCacheTest(unittest.TestCase):
    def test_explicit_cache_dir_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            path = resolve_longcat_cache_dir(root / "explicit")

            self.assertEqual(path, root / "explicit")

    def test_longcat_env_wins_over_global_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                ANYTRAIN_LONGCAT_CACHE_ENV: str(root / "longcat"),
                ANYTRAIN_CACHE_ENV: str(root / "anytrain"),
            }

            with patch.dict(os.environ, env, clear=False):
                path = resolve_longcat_cache_dir()

            self.assertEqual(path, root / "longcat")

    def test_global_env_is_used_when_longcat_env_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {ANYTRAIN_CACHE_ENV: str(root / "anytrain")}

            with patch.dict(os.environ, env, clear=False):
                os.environ.pop(ANYTRAIN_LONGCAT_CACHE_ENV, None)
                path = resolve_longcat_cache_dir()

            self.assertEqual(path, root / "anytrain" / "longcat-audio-codec")

    def test_empty_env_fails(self):
        with (
            patch.dict(os.environ, {ANYTRAIN_LONGCAT_CACHE_ENV: ""}, clear=False),
            self.assertRaisesRegex(ValueError, ANYTRAIN_LONGCAT_CACHE_ENV),
        ):
            resolve_longcat_cache_dir()


if __name__ == "__main__":
    unittest.main()
