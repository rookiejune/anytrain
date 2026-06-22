import unittest


class TinyRegressionExampleTest(unittest.TestCase):
    def test_run_tiny_regression(self):
        from examples.tiny_regression import run_tiny_regression

        trainer, module = run_tiny_regression(
            num_samples=8,
            batch_size=4,
            enable_progress_bar=False,
        )

        self.assertEqual(trainer.current_epoch, 1)
        self.assertEqual(str(trainer.default_root_dir), "outputs/anytrain/tiny_regression")
        self.assertEqual(module.model.in_features, 4)
        self.assertIsNotNone(module.optimizer)


if __name__ == "__main__":
    unittest.main()
