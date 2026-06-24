import unittest

from anytrain.idspace import IdSpace, Modality, ModalityBlock


class IdSpaceTest(unittest.TestCase):
    def test_space_keeps_special_token_ids_separate_from_modalities(self):
        space = IdSpace(
            {"pad": 0, "bos": 1, "eos": 2},
            [ModalityBlock(Modality.TEXT, 3, 4), ModalityBlock(Modality.AUDIO, 7, 2)],
        )

        self.assertEqual(space.special_token_ids, {"pad": 0, "bos": 1, "eos": 2})
        self.assertEqual(space.special_token_id("pad"), 0)
        self.assertEqual(space.special_token_id("eos"), 2)
        self.assertEqual(space.to_global(Modality.TEXT, [0, 3]), [3, 6])
        self.assertEqual(space.to_global(Modality.AUDIO, [0, 1]), [7, 8])
        self.assertEqual(space.to_local(Modality.TEXT, [3, 6]), [0, 3])
        self.assertEqual(space.vocab_size, 9)
        with self.assertRaises(TypeError):
            space.special_token_ids["pad"] = 9

    def test_space_rejects_special_token_ids_inside_modality_decode(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "special"):
            space.to_local(Modality.TEXT, [0])
        self.assertEqual(space.to_local(Modality.TEXT, [0, 1], skip_special=True), [0])

    def test_space_allows_sparse_special_token_ids_inside_modality_block(self):
        space = IdSpace(
            {"pad": 0, "bos": 3, "eos": 5},
            [ModalityBlock(Modality.TEXT, 0, 6)],
        )

        self.assertEqual(space.special_token_id("bos"), 3)
        self.assertEqual(space.all_special_ids, (0, 3, 5))
        self.assertEqual(space.regular_blocks(Modality.TEXT), ((1, 2), (4, 1)))
        self.assertEqual(space.to_global(Modality.TEXT, [1, 2, 4]), [1, 2, 4])
        with self.assertRaisesRegex(ValueError, "special"):
            space.to_global(Modality.TEXT, [3])
        with self.assertRaisesRegex(ValueError, "special"):
            space.to_local(Modality.TEXT, [3])
        self.assertEqual(space.to_local(Modality.TEXT, [0, 1, 3, 4, 5], skip_special=True), [1, 4])
        self.assertEqual(space.vocab_size, 6)

    def test_rejects_implicit_space_inputs(self):
        with self.assertRaisesRegex(TypeError, "special_token_ids"):
            IdSpace(["pad"], [ModalityBlock(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(TypeError, "ModalityBlock"):
            IdSpace({"pad": 0}, [(Modality.TEXT, 2)])

    def test_rejects_raw_string_modality(self):
        with self.assertRaisesRegex(TypeError, "Modality"):
            ModalityBlock("text", 0, 2)


if __name__ == "__main__":
    unittest.main()
