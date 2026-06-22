import unittest

from anytrain.registry import Registry
from anytrain.types import AutoNameEnum


class RegistryTest(unittest.TestCase):
    def test_auto_name_enum(self):
        class LocalName(AutoNameEnum):
            SOME_VALUE = "some_value"

        self.assertEqual(LocalName.SOME_VALUE.value, "some_value")

    def test_register_and_create(self):
        registry = Registry[str, type]()
        registry.register("int", int)

        self.assertIn("int", registry)
        self.assertEqual(registry.create("int", "3"), 3)

    def test_duplicate_key_fails(self):
        registry = Registry[str, type]({"int": int})

        with self.assertRaises(KeyError):
            registry.register("int", float)


if __name__ == "__main__":
    unittest.main()
