import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from anytrain.codec.unicodec.assets import UniCodecAssets
from anytrain.codec.unicodec.codec import UniCodec
from torch import nn


class UniCodecCodecTest(unittest.TestCase):
    def test_from_pretrained_loads_model_from_resolved_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets: UniCodecAssets = {
                "cache_dir": root,
                "config": root / "config.yaml",
                "checkpoint": root / "unicode.ckpt",
            }

            with (
                patch(
                    "anytrain.codec.unicodec.codec.ensure_unicodec_assets",
                    return_value=assets,
                ),
                patch(
                    "anytrain.codec.unicodec.codec._load_unicodec_model",
                    return_value=FakeUniCodecFactory,
                ),
            ):
                codec = UniCodec.from_pretrained(device="cpu")

            self.assertIsInstance(codec.model, FakeUniCodecModel)
            self.assertEqual(FakeUniCodecFactory.config_path, str(assets["config"]))
            self.assertEqual(FakeUniCodecFactory.model_path, str(assets["checkpoint"]))
            self.assertEqual(codec.device, torch.device("cpu"))
            self.assertEqual(codec.assets, assets)
            self.assertTrue(codec.model.eval_called)

    def test_encode_returns_codes_and_passes_domain_contract(self):
        model = FakeUniCodecModel()
        codec = UniCodec(model=model, device=torch.device("cpu"), assets=_assets())
        audio = torch.zeros((2, 8))

        codes = codec.encode(audio, domain=(0, "2"), bandwidth_id=3)

        self.assertEqual(tuple(codes.shape), (1, 2, 4))
        self.assertEqual(model.domains, ("0", "2"))
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([3])))

    def test_encode_features_exposes_upstream_feature_boundary(self):
        model = FakeUniCodecModel()
        codec = UniCodec(model=model, device=torch.device("cpu"), assets=_assets())
        audio = torch.zeros((2, 8))

        features, codes = codec.encode_features(audio, domain=(0, "2"), bandwidth_id=3)

        self.assertEqual(tuple(features.shape), (2, 3, 4))
        self.assertEqual(tuple(codes.shape), (1, 2, 4))
        self.assertEqual(model.domains, ("0", "2"))
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([3])))

    def test_reconstruct_decodes_encoded_codes(self):
        model = FakeUniCodecModel()
        codec = UniCodec(model=model, device=torch.device("cpu"), assets=_assets())

        audio = codec.reconstruct(torch.zeros((1, 8)), domain="1")

        self.assertEqual(tuple(audio.shape), (1, 16))
        self.assertEqual(model.domains, ("1",))
        self.assertTrue(model.decode_called)
        self.assertTrue(model.codes_to_features_called)

    def test_decode_converts_codes_before_decode(self):
        model = FakeUniCodecModel()
        codec = UniCodec(model=model, device=torch.device("cpu"), assets=_assets())
        codes = torch.ones((1, 2, 4), dtype=torch.long)

        audio = codec.decode(codes, bandwidth_id=torch.tensor([2]))

        self.assertEqual(tuple(audio.shape), (2, 16))
        self.assertTrue(model.codes_to_features_called)
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([2])))

    def test_decode_features_uses_continuous_feature_boundary(self):
        model = FakeUniCodecModel()
        codec = UniCodec(model=model, device=torch.device("cpu"), assets=_assets())
        features = torch.ones((2, 3, 4))

        audio = codec.decode_features(features, bandwidth_id=torch.tensor([2]))

        self.assertEqual(tuple(audio.shape), (2, 16))
        self.assertTrue(model.decode_called)
        self.assertFalse(model.codes_to_features_called)
        self.assertTrue(torch.equal(model.bandwidth_id, torch.tensor([2])))

    def test_encode_rejects_unknown_domain(self):
        codec = UniCodec(
            model=FakeUniCodecModel(),
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "domain"):
            codec.encode(torch.zeros((1, 8)), domain="speech")

    def test_encode_rejects_domain_batch_mismatch(self):
        codec = UniCodec(
            model=FakeUniCodecModel(),
            device=torch.device("cpu"),
            assets=_assets(),
        )

        with self.assertRaisesRegex(ValueError, "batch size"):
            codec.encode(torch.zeros((2, 8)), domain=("0",))


class FakeUniCodecFactory:
    config_path: str | None = None
    model_path: str | None = None

    @classmethod
    def from_pretrained0802(cls, config_path, model_path):
        cls.config_path = config_path
        cls.model_path = model_path
        return FakeUniCodecModel()


class FakeUniCodecModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.domains: tuple[str, ...] | None = None
        self.bandwidth_id: torch.Tensor | None = None
        self.decode_called = False
        self.codes_to_features_called = False
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return super().eval()

    def encode_infer(self, audio: torch.Tensor, domain, *, bandwidth_id: torch.Tensor):
        self.domains = tuple(domain)
        self.bandwidth_id = bandwidth_id
        batch = audio.size(0)
        return torch.ones((batch, 3, 4)), torch.ones((1, batch, 4), dtype=torch.long)

    def decode(self, features: torch.Tensor, *, bandwidth_id: torch.Tensor):
        self.decode_called = True
        self.bandwidth_id = bandwidth_id
        return torch.ones((features.size(0), 16))

    def codes_to_features(self, codes: torch.Tensor):
        self.codes_to_features_called = True
        return torch.ones((codes.size(1), 3, codes.size(2)))


def _assets() -> UniCodecAssets:
    return {
        "cache_dir": Path(),
        "config": Path(),
        "checkpoint": Path(),
    }


if __name__ == "__main__":
    unittest.main()
