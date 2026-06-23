import unittest

from anytrain.idspace import Modality, ModalityRange, MultiTokenizer, SubTokenizer, TokenLayout


class FakeTokenizer:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def encode(self, value: object) -> list[int]:
        return self.ids

    def decode(self, ids: list[int]) -> tuple[int, ...]:
        return tuple(ids)


class TokenLayoutTest(unittest.TestCase):
    def test_layout_keeps_special_token_ids_separate_from_modalities(self):
        layout = TokenLayout(
            {"pad": 0, "bos": 1, "eos": 2},
            [ModalityRange(Modality.TEXT, 3, 4), ModalityRange(Modality.AUDIO, 7, 2)],
        )

        self.assertEqual(layout.special_token_id("pad"), 0)
        self.assertEqual(layout.special_token_id("eos"), 2)
        self.assertEqual(layout.to_global(Modality.TEXT, [0, 3]), [3, 6])
        self.assertEqual(layout.to_global(Modality.AUDIO, [0, 1]), [7, 8])
        self.assertEqual(layout.to_local(Modality.TEXT, [3, 6]), [0, 3])
        self.assertEqual(layout.vocab_size, 9)

    def test_layout_rejects_special_token_ids_inside_modality_decode(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "special"):
            layout.to_local(Modality.TEXT, [0])
        self.assertEqual(layout.to_local(Modality.TEXT, [0, 1], skip_special=True), [0])

    def test_layout_allows_sparse_special_token_ids_inside_modality_range(self):
        layout = TokenLayout(
            {"pad": 0, "bos": 3, "eos": 5},
            [ModalityRange(Modality.TEXT, 0, 6)],
        )

        self.assertEqual(layout.special_token_id("bos"), 3)
        self.assertEqual(layout.to_global(Modality.TEXT, [1, 2, 4]), [1, 2, 4])
        with self.assertRaisesRegex(ValueError, "special"):
            layout.to_global(Modality.TEXT, [3])
        with self.assertRaisesRegex(ValueError, "special"):
            layout.to_local(Modality.TEXT, [3])
        self.assertEqual(layout.to_local(Modality.TEXT, [0, 1, 3, 4, 5], skip_special=True), [1, 4])
        self.assertEqual(layout.vocab_size, 6)

    def test_multi_tokenizer_uses_modality_local_ids(self):
        tokenizer = MultiTokenizer(
            {
                Modality.TEXT: FakeTokenizer([0, 2]),
                Modality.AUDIO: FakeTokenizer([1]),
            },
        )

        self.assertEqual(tokenizer.encode(Modality.TEXT, "x"), [0, 2])
        self.assertEqual(tokenizer.decode(Modality.TEXT, [0, 2]), (0, 2))
        self.assertEqual(tokenizer.encode(Modality.AUDIO, "x"), [1])
        self.assertEqual(tokenizer.decode(Modality.AUDIO, [1]), (1,))

    def test_rejects_implicit_layout_inputs(self):
        with self.assertRaisesRegex(TypeError, "special_token_ids"):
            TokenLayout(["pad"], [ModalityRange(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(TypeError, "ModalityRange"):
            TokenLayout({"pad": 0}, [(Modality.TEXT, 2)])

    def test_rejects_raw_string_modality(self):
        with self.assertRaisesRegex(TypeError, "Modality"):
            ModalityRange("text", 0, 2)

        with self.assertRaisesRegex(TypeError, "Modality"):
            MultiTokenizer({"text": FakeTokenizer([])})

    def test_rejects_objects_without_tokenizer_protocol(self):
        with self.assertRaisesRegex(TypeError, "SubTokenizer"):
            MultiTokenizer({Modality.TEXT: object()})

    def test_sub_tokenizer_is_protocol(self):
        self.assertIsInstance(FakeTokenizer([]), SubTokenizer)


if __name__ == "__main__":
    unittest.main()
