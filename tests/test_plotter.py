import builtins
import importlib.util
import unittest

import torch

from anytrain.plotter import Plotter, TensorImagePlotter

MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None


class PlotterTest(unittest.TestCase):
    def test_plotter_import_does_not_require_matplotlib(self):
        self.assertIn("TensorImagePlotter", TensorImagePlotter.__name__)
        self.assertIsNotNone(Plotter)

    def test_tensor_image_plotter_reports_missing_matplotlib(self):
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("matplotlib"):
                raise ImportError("missing matplotlib")
            return original_import(name, globals, locals, fromlist, level)

        with (
            unittest.mock.patch("builtins.__import__", side_effect=fake_import),
            self.assertRaisesRegex(ImportError, r"anytrain\[plot\]"),
        ):
            TensorImagePlotter()(torch.zeros(2, 2))

    @unittest.skipUnless(MATPLOTLIB_AVAILABLE, "matplotlib is not installed")
    def test_tensor_image_plotter_returns_matplotlib_figure(self):
        from matplotlib import pyplot
        from matplotlib.figure import Figure

        figure = TensorImagePlotter(title="sample")(torch.zeros(3, 4, 5))

        self.assertIsInstance(figure, Figure)
        self.assertEqual(figure.axes[0].get_title(), "sample")
        pyplot.close(figure)

    def test_tensor_image_plotter_rejects_batched_images(self):
        with self.assertRaisesRegex(ValueError, "single"):
            TensorImagePlotter()(torch.zeros(2, 3, 4, 5))

    def test_tensor_image_plotter_rejects_ambiguous_channel_shape(self):
        with self.assertRaisesRegex(ValueError, "CHW or HWC"):
            TensorImagePlotter()(torch.zeros(5, 6, 7))


if __name__ == "__main__":
    unittest.main()
