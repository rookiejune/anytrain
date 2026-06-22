import unittest

import torch
from torch import nn

from anytrain.utils.optim import (
    DEFAULT_OUTPUT_HEAD_MODULE_NAMES,
    is_default_muon_parameter,
    split_muon_params,
)


def _param_ids(params: list[nn.Parameter]) -> set[int]:
    return {id(param) for param in params}


class SplitMuonParamsTest(unittest.TestCase):
    def test_split_muon_params_uses_2d_weight_rule(self):
        model = nn.Sequential(
            nn.Linear(4, 3),
            nn.LayerNorm(3),
        )

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model[0].weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model[0].bias), id(model[1].weight), id(model[1].bias)},
        )

    def test_split_muon_params_excludes_embedding_weights_by_default(self):
        model = nn.Sequential(
            nn.Embedding(8, 3),
            nn.Linear(3, 4),
        )

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model[1].weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model[0].weight), id(model[1].bias)},
        )

    def test_split_muon_params_excludes_output_head_weights_by_default(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.proj.weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.proj.bias), id(model.lm_head.weight)},
        )

    def test_split_muon_params_excludes_tied_embedding_head_weight(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(8, 4)
                self.proj = nn.Linear(4, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)
                self.lm_head.weight = self.embed_tokens.weight

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.proj.weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.embed_tokens.weight), id(model.proj.bias)},
        )

    def test_split_muon_params_excludes_custom_norm_class_by_default(self):
        class TinyRMSNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3, 3))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3, 3, bias=False)
                self.rms_norm = TinyRMSNorm()

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.proj.weight)})
        self.assertEqual(_param_ids(non_muon_parameters), {id(model.rms_norm.weight)})

    def test_split_muon_params_can_override_output_head_names(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(
            model,
            output_head_module_names=frozenset(),
        )

        self.assertEqual(_param_ids(muon_parameters), {id(model.lm_head.weight)})
        self.assertEqual(non_muon_parameters, [])

    def test_split_muon_params_skips_frozen_params_by_default(self):
        model = nn.Linear(4, 3, bias=False)
        model.weight.requires_grad_(False)

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(muon_parameters, [])
        self.assertEqual(non_muon_parameters, [])

    def test_split_muon_params_can_include_frozen_params(self):
        model = nn.Linear(4, 3, bias=False)
        model.weight.requires_grad_(False)

        muon_parameters, non_muon_parameters = split_muon_params(model, requires_grad_only=False)

        self.assertEqual(_param_ids(muon_parameters), {id(model.weight)})
        self.assertEqual(non_muon_parameters, [])

    def test_split_muon_params_accepts_custom_predicate(self):
        model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Embedding(8, 3),
        )

        def is_embedding_parameter(name: str, parameter: nn.Parameter) -> bool:
            return name == "1.weight"

        muon_parameters, non_muon_parameters = split_muon_params(
            model,
            is_muon_parameter=is_embedding_parameter,
        )

        self.assertEqual(_param_ids(muon_parameters), {id(model[1].weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model[0].weight), id(model[0].bias)},
        )

    def test_default_muon_predicate_matches_2d_weight(self):
        weight = nn.Parameter(torch.randn(3, 4))
        bias = nn.Parameter(torch.randn(3))
        embedding = nn.Embedding(8, 4)

        self.assertTrue(is_default_muon_parameter("linear.weight", weight))
        self.assertFalse(is_default_muon_parameter("linear.bias", bias))
        self.assertFalse(
            is_default_muon_parameter(
                "embedding.weight",
                embedding.weight,
                module=embedding,
            )
        )
        self.assertFalse(
            is_default_muon_parameter(
                "lm_head.weight",
                weight,
                output_head_module_names=DEFAULT_OUTPUT_HEAD_MODULE_NAMES,
            )
        )

    def test_split_muon_params_rejects_non_module(self):
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            split_muon_params(torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
