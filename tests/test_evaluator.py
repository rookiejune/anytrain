import unittest

import torch
from torch import nn

from anytrain.evaluator import EvaluatorABC, EvaluatorGroup


class ErrorEvaluator(EvaluatorABC):
    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
        error = prediction - target
        return {
            "mae": error.abs().mean(),
            "mse": error.pow(2).mean(),
        }


class FloatEvaluator(EvaluatorABC):
    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
        return {"score": float((prediction - target).abs().mean())}


class IntegerValueEvaluator(EvaluatorABC):
    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
        return {"score": 1}


class VectorTensorEvaluator(EvaluatorABC):
    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
        return {"score": (prediction - target).abs()}


class SeparatorKeyEvaluator(EvaluatorABC):
    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
        return {"a/b": (prediction - target).abs().mean()}


class RunningAbsoluteError(EvaluatorABC):
    def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
        return {"mae": (prediction - target).abs().mean()}


class EvaluatorTest(unittest.TestCase):
    def test_evaluator_group_forward_combines_metric_dicts_without_storing(self):
        prediction = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        evaluator = EvaluatorGroup({"error": ErrorEvaluator()})

        metrics = evaluator(prediction, target)

        self.assertEqual(set(metrics), {"error/mae", "error/mse"})
        self.assertTrue(torch.isclose(metrics["error/mae"], torch.tensor(1.5)))
        self.assertTrue(metrics["error/mae"].requires_grad)
        with self.assertRaisesRegex(ValueError, "No metric values"):
            evaluator.compute()

    def test_evaluator_group_update_stores_detached_0d_tensors(self):
        prediction = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        evaluator = EvaluatorGroup({"error": ErrorEvaluator()})

        evaluator.update(prediction, target)
        metrics = evaluator.compute()

        self.assertFalse(metrics["error/mae"].requires_grad)

    def test_evaluator_group_accepts_float_metric_values(self):
        evaluator = EvaluatorGroup({"float_score": FloatEvaluator()})

        metrics = evaluator(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))

        self.assertEqual(metrics["float_score/score"], 1.5)

    def test_evaluator_group_registers_metrics_in_module_dict(self):
        evaluator = EvaluatorGroup({"error": ErrorEvaluator(), "running": RunningAbsoluteError()})

        self.assertEqual(set(evaluator.metrics.keys()), {"error", "running"})
        self.assertIsInstance(evaluator.metrics, nn.ModuleDict)

    def test_evaluator_abc_can_return_direct_evaluate_without_storing(self):
        prediction = torch.tensor([1.0, 3.0], requires_grad=True)
        target = torch.tensor([0.0, 1.0])
        evaluator = ErrorEvaluator()

        metrics = evaluator.evaluate(prediction, target)

        self.assertTrue(metrics["mae"].requires_grad)

    def test_evaluator_call_returns_validated_metric_dict_without_storing(self):
        evaluator = RunningAbsoluteError()

        metrics = evaluator(torch.tensor([1.0]), torch.tensor([0.0]))

        self.assertTrue(torch.isclose(metrics["mae"], torch.tensor(1.0)))
        with self.assertRaisesRegex(ValueError, "No metric values"):
            evaluator.compute()

    def test_evaluator_abc_rejects_separator_in_metric_keys(self):
        evaluator = SeparatorKeyEvaluator()

        with self.assertRaisesRegex(ValueError, "separator"):
            evaluator(torch.tensor([1.0]), torch.tensor([0.0]))

        with self.assertRaisesRegex(ValueError, "separator"):
            evaluator.update(torch.tensor([1.0]), torch.tensor([0.0]))

    def test_evaluator_group_supports_stateful_lifecycle(self):
        evaluator = EvaluatorGroup({"error": RunningAbsoluteError()})

        evaluator.update(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 1.0]))
        evaluator.update(torch.tensor([2.0]), torch.tensor([1.0]))
        metrics = evaluator.compute()
        evaluator.reset()

        self.assertTrue(torch.isclose(metrics["error/mae"], torch.tensor(1.25)))
        with self.assertRaisesRegex(ValueError, "No metric values"):
            evaluator.compute()

    def test_evaluator_group_rejects_integer_metric_values(self):
        evaluator = EvaluatorGroup({"invalid": IntegerValueEvaluator()})

        with self.assertRaisesRegex(TypeError, "float or 0-d tensor"):
            evaluator(torch.tensor([1.0]), torch.tensor([0.0]))

    def test_evaluator_group_rejects_non_scalar_tensor_metric_values(self):
        evaluator = EvaluatorGroup({"invalid": VectorTensorEvaluator()})

        with self.assertRaisesRegex(ValueError, "0-d tensor"):
            evaluator(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0]))

    def test_evaluator_group_rejects_separator_in_metric_names(self):
        with self.assertRaisesRegex(ValueError, "separator"):
            EvaluatorGroup({"metric/a": ErrorEvaluator()})

    def test_evaluator_group_rejects_non_string_metric_names(self):
        with self.assertRaisesRegex(TypeError, "string"):
            EvaluatorGroup({1: ErrorEvaluator()})

    def test_evaluator_group_uses_moduledict_name_rules(self):
        with self.assertRaisesRegex(KeyError, "attribute"):
            EvaluatorGroup({"float": FloatEvaluator()})

    def test_evaluator_group_rejects_separator_in_metric_keys(self):
        evaluator = EvaluatorGroup({"metric": SeparatorKeyEvaluator()})

        with self.assertRaisesRegex(ValueError, "separator"):
            evaluator(torch.tensor([1.0]), torch.tensor([0.0]))

    def test_evaluator_group_rejects_empty_metrics(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            EvaluatorGroup({})

    def test_evaluator_group_rejects_metric_sequences(self):
        with self.assertRaisesRegex(TypeError, "mapping"):
            EvaluatorGroup([ErrorEvaluator()])

    def test_evaluator_group_requires_evaluator_abc(self):
        class PlainEvaluator(nn.Module):
            def evaluate(self, prediction: torch.Tensor, target: torch.Tensor):
                return {"mae": (prediction - target).abs().mean()}

        with self.assertRaisesRegex(TypeError, "EvaluatorABC"):
            EvaluatorGroup({"plain": PlainEvaluator()})


if __name__ == "__main__":
    unittest.main()
