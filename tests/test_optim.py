import unittest
from tempfile import TemporaryDirectory

import torch
from lightning.pytorch import LightningModule, Trainer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from anytrain.optim import (
    AdamWConfig,
    AdamWDecayPolicy,
    CompositeOptimizer,
    CurveShape,
    LLMLightningOptimizerConfig,
    LLMOptimizationConfig,
    MuonAdamWConfig,
    MuonAdjustLRFn,
    MuonConfig,
    SchedulerConfig,
    SchedulerPhaseConfig,
    create_adamw_optimizer,
    create_llm_lightning_optimizers,
    create_llm_optimizer,
    create_muon_adamw_optimizer,
    create_scheduler,
    make_scheduler_config,
    split_adamw_decay_params,
    split_muon_params,
)
from anytrain.optim import llm as llm_optim


def _param_ids(params: list[nn.Parameter]) -> set[int]:
    return {id(param) for param in params}


def _group_param_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        id(param)
        for group in optimizer.param_groups
        for param in group["params"]
    }


class TinyLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)


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

    def test_split_muon_params_keeps_module_biases_non_muon(self):
        class ModuleWithMatrixBias(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 4))
                self.bias = nn.Parameter(torch.randn(3, 4))

        model = ModuleWithMatrixBias()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.weight)})
        self.assertEqual(_param_ids(non_muon_parameters), {id(model.bias)})

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

    def test_split_muon_params_uses_embedding_type_not_name(self):
        class FakeEmbedding(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(8, 3))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = FakeEmbedding()

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(_param_ids(muon_parameters), {id(model.embed_tokens.weight)})
        self.assertEqual(non_muon_parameters, [])

    def test_split_muon_params_excludes_custom_module_types(self):
        class Adapter(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4, bias=False)
                self.adapter = Adapter()

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(
            model,
            excluded_module_types=(Adapter,),
        )

        self.assertEqual(_param_ids(muon_parameters), {id(model.proj.weight)})
        self.assertEqual(_param_ids(non_muon_parameters), {id(model.adapter.weight)})

    def test_split_muon_params_includes_head_named_weights_by_default(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(
            _param_ids(muon_parameters),
            {id(model.proj.weight), id(model.lm_head.weight)},
        )
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.proj.bias)},
        )

    def test_split_muon_params_excludes_explicit_modules(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 4)
                self.lm_head = nn.Linear(4, 8, bias=False)

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(
            model,
            excluded_modules=(model.lm_head,),
        )

        self.assertEqual(
            _param_ids(muon_parameters),
            {id(model.proj.weight)},
        )
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.proj.bias), id(model.lm_head.weight)},
        )

    def test_split_muon_params_excludes_explicit_module_subtree(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.decoder = nn.Linear(4, 8, bias=False)
                self.output = nn.Sequential(
                    nn.Linear(4, 4, bias=False),
                    nn.Linear(4, 8, bias=False),
                )

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(
            model,
            excluded_modules=(model.output,),
        )

        self.assertEqual(_param_ids(muon_parameters), {id(model.decoder.weight)})
        self.assertEqual(
            _param_ids(non_muon_parameters),
            {id(model.output[0].weight), id(model.output[1].weight)},
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

    def test_split_muon_params_uses_non_muon_for_any_shared_parameter_owner(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(4, 8, bias=False)
                self.embed_tokens = nn.Embedding(8, 4)
                self.proj.weight = self.embed_tokens.weight

        model = Model()

        muon_parameters, non_muon_parameters = split_muon_params(model)

        self.assertEqual(muon_parameters, [])
        self.assertEqual(_param_ids(non_muon_parameters), {id(model.embed_tokens.weight)})

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

    def test_split_muon_params_rejects_non_module(self):
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            split_muon_params(torch.tensor(1.0))

    def test_split_muon_params_rejects_excluded_module_outside_root(self):
        with self.assertRaisesRegex(ValueError, "belong"):
            split_muon_params(nn.Linear(4, 4), excluded_modules=(nn.Linear(4, 4),))


class AdamWOptimizerTest(unittest.TestCase):
    def test_split_adamw_decay_params_excludes_norm_embedding_and_bias(self):
        model = TinyLLM()

        decay_params, no_decay_params = split_adamw_decay_params(model)

        self.assertEqual(_param_ids(decay_params), {id(model.proj.weight), id(model.lm_head.weight)})
        self.assertEqual(
            _param_ids(no_decay_params),
            {
                id(model.embed_tokens.weight),
                id(model.proj.bias),
                id(model.norm.weight),
                id(model.norm.bias),
            },
        )

    def test_adamw_decay_policy_type_is_exported(self):
        policy = AdamWDecayPolicy.STANDARD

        self.assertEqual(policy.value, "standard")

    def test_split_adamw_decay_params_excludes_explicit_modules(self):
        model = TinyLLM()

        decay_params, no_decay_params = split_adamw_decay_params(
            model,
            excluded_modules=(model.lm_head,),
        )

        self.assertEqual(_param_ids(decay_params), {id(model.proj.weight)})
        self.assertEqual(
            _param_ids(no_decay_params),
            {
                id(model.embed_tokens.weight),
                id(model.proj.bias),
                id(model.norm.weight),
                id(model.norm.bias),
                id(model.lm_head.weight),
            },
        )

    def test_split_adamw_decay_params_decays_standard_weight_parameters_by_default(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(2, 4, kernel_size=3, bias=False)
                self.proj = nn.Linear(4, 4, bias=False)

        model = Model()

        decay_params, no_decay_params = split_adamw_decay_params(model)

        self.assertEqual(_param_ids(decay_params), {id(model.conv.weight), id(model.proj.weight)})
        self.assertEqual(_param_ids(no_decay_params), set())

    def test_split_adamw_decay_params_can_use_muon_eligible_policy(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(2, 4, kernel_size=3, bias=False)
                self.proj = nn.Linear(4, 4, bias=False)

        model = Model()

        decay_params, no_decay_params = split_adamw_decay_params(
            model,
            decay_policy=AdamWDecayPolicy.MUON_ELIGIBLE,
        )

        self.assertEqual(_param_ids(decay_params), {id(model.proj.weight)})
        self.assertEqual(_param_ids(no_decay_params), {id(model.conv.weight)})

    def test_split_adamw_decay_params_accepts_decay_policy_string(self):
        model = nn.Linear(4, 4, bias=False)

        decay_params, no_decay_params = split_adamw_decay_params(
            model,
            decay_policy="standard",
        )

        self.assertEqual(_param_ids(decay_params), {id(model.weight)})
        self.assertEqual(no_decay_params, [])

    def test_create_adamw_optimizer_uses_decay_and_no_decay_groups(self):
        model = TinyLLM()
        optimizer = create_adamw_optimizer(model, AdamWConfig(lr=3e-4, weight_decay=0.1))

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.1, 0.0])
        self.assertEqual(_group_param_ids(optimizer), {id(param) for param in model.parameters()})

    def test_create_adamw_optimizer_can_disable_decay_for_selected_params(self):
        model = TinyLLM()
        optimizer = create_adamw_optimizer(
            model,
            AdamWConfig(lr=3e-4, weight_decay=0.1),
            selected_params=[model.proj.weight],
            decay_selected_params=False,
        )

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(_group_param_ids(optimizer), {id(model.proj.weight)})

    def test_split_adamw_decay_params_rejects_selected_param_outside_module(self):
        with self.assertRaisesRegex(ValueError, "belong"):
            split_adamw_decay_params(
                nn.Linear(4, 4),
                selected_params=[nn.Parameter(torch.randn(4, 4))],
            )

    def test_split_adamw_decay_params_rejects_invalid_decay_policy(self):
        with self.assertRaisesRegex(ValueError, "decay_policy"):
            split_adamw_decay_params(nn.Linear(4, 4), decay_policy="hidden")


class MuonAdamWOptimizerTest(unittest.TestCase):
    def test_create_muon_adamw_optimizer_returns_composite_optimizer(self):
        model = TinyLLM()

        optimizer = create_muon_adamw_optimizer(
            model,
            muon=MuonConfig(lr=3e-4),
            adamw=AdamWConfig(lr=3e-4),
            excluded_modules=(model.lm_head,),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertEqual(set(optimizer.optimizers), {"muon", "adamw"})
        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(
            optimizer.optimizers["muon"].param_groups[0]["adjust_lr_fn"],
            MuonAdjustLRFn.MATCH_RMS_ADAMW,
        )
        self.assertEqual(_group_param_ids(optimizer), {id(param) for param in model.parameters()})
        self.assertNotIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["muon"]))
        self.assertIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["adamw"]))

    def test_create_muon_adamw_optimizer_uses_no_decay_for_adamw_fallback(self):
        model = TinyLLM()

        optimizer = create_muon_adamw_optimizer(
            model,
            muon=MuonConfig(lr=3e-4, weight_decay=0.1),
            adamw=AdamWConfig(lr=3e-4, weight_decay=0.1),
            excluded_modules=(model.lm_head,),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        adamw_optimizer = optimizer.optimizers["adamw"]
        self.assertTrue(
            all(group["weight_decay"] == 0.0 for group in adamw_optimizer.param_groups)
        )

    def test_composite_optimizer_scheduler_updates_child_param_groups(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=MuonConfig(lr=1.0),
            adamw=AdamWConfig(lr=1.0),
            excluded_modules=(model.lm_head,),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.5)

        optimizer.step()
        scheduler.step()

        self.assertTrue(all(group["lr"] == 0.5 for group in optimizer.param_groups))
        self.assertTrue(
            all(group["lr"] == 0.5 for group in optimizer.optimizers["muon"].param_groups)
        )
        self.assertTrue(
            all(group["lr"] == 0.5 for group in optimizer.optimizers["adamw"].param_groups)
        )

    def test_composite_optimizer_state_dict_round_trip_uses_optimizer_names(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=MuonConfig(lr=3e-4),
            adamw=AdamWConfig(lr=3e-4),
            excluded_modules=(model.lm_head,),
        )

        state = optimizer.state_dict()

        self.assertEqual(set(state["optimizers"]), {"muon", "adamw"})
        optimizer.load_state_dict(state)

        first_child_group = optimizer.optimizers["muon"].param_groups[0]
        self.assertIs(optimizer.param_groups[0], first_child_group)

    def test_composite_optimizer_rejects_mismatched_state_names(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=MuonConfig(lr=3e-4),
            adamw=AdamWConfig(lr=3e-4),
            excluded_modules=(model.lm_head,),
        )

        with self.assertRaisesRegex(ValueError, "state names"):
            optimizer.load_state_dict({"optimizers": {"muon": {}}})

    def test_composite_optimizer_rejects_shared_child_parameters(self):
        model = nn.Linear(4, 4)
        optimizer_a = torch.optim.AdamW([model.weight], lr=1e-3)
        optimizer_b = torch.optim.SGD([model.weight], lr=1e-3)

        with self.assertRaisesRegex(ValueError, "share parameters"):
            CompositeOptimizer({"adamw": optimizer_a, "sgd": optimizer_b})

    def test_composite_optimizer_rejects_add_param_group_after_composition(self):
        model = nn.Linear(4, 4)
        optimizer = CompositeOptimizer({"adamw": torch.optim.AdamW([model.weight], lr=1e-3)})

        with self.assertRaisesRegex(RuntimeError, "add_param_group"):
            optimizer.add_param_group({"params": [model.bias]})


class LLMOptimizerTest(unittest.TestCase):
    def test_llm_config_from_preset_uses_preset_adamw_defaults(self):
        config = LLMOptimizationConfig.from_preset("sft")

        self.assertIsInstance(config.optimizer_config, AdamWConfig)
        self.assertEqual(config.optimizer_config.lr, 2e-5)
        self.assertEqual(config.optimizer_config.weight_decay, 0.01)
        self.assertEqual(config.optimizer_config.betas, (0.9, 0.999))

    def test_llm_config_from_preset_uses_cpt_defaults(self):
        config = LLMOptimizationConfig.from_preset("cpt").optimizer_config

        self.assertIsInstance(config, AdamWConfig)
        self.assertEqual(config.lr, 5e-5)
        self.assertEqual(config.weight_decay, 0.1)

    def test_llm_config_from_preset_uses_optimizer_string(self):
        config = LLMOptimizationConfig.from_preset("pretrain", optimizer="muon")

        self.assertIsInstance(config.optimizer_config, MuonAdamWConfig)
        self.assertEqual(config.optimizer_config.adamw.lr, 3e-4)
        self.assertEqual(config.optimizer_config.muon.lr, 3e-4)
        self.assertEqual(config.optimizer_config.adamw.weight_decay, 0.1)
        self.assertEqual(config.optimizer_config.muon.weight_decay, 0.1)

    def test_llm_config_from_preset_rejects_unknown_optimizer_string(self):
        with self.assertRaisesRegex(ValueError, "optimizer"):
            LLMOptimizationConfig.from_preset("pretrain", optimizer="sgd")

    def test_llm_config_from_preset_rejects_unknown_preset_string(self):
        with self.assertRaisesRegex(ValueError, "preset"):
            LLMOptimizationConfig.from_preset("dpo")

    def test_llm_config_rejects_invalid_optimizer_config(self):
        with self.assertRaisesRegex(TypeError, "optimizer_config"):
            LLMOptimizationConfig(optimizer_config="adamw")

    def test_create_llm_optimizer_uses_adamw_for_adamw_config(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            LLMOptimizationConfig(optimizer_config=AdamWConfig(lr=1e-4)),
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_create_llm_optimizer_uses_composite_for_muon_config(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            LLMOptimizationConfig.from_preset("pretrain", optimizer="muon"),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)

    def test_create_llm_optimizer_passes_excluded_modules(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            LLMOptimizationConfig.from_preset(
                "pretrain",
                optimizer="muon",
                excluded_modules=(model.lm_head,),
            ),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        self.assertNotIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["muon"]))
        self.assertIn(id(model.lm_head.weight), _group_param_ids(optimizer.optimizers["adamw"]))

    def test_llm_config_accepts_custom_muon_adamw_config(self):
        adamw = AdamWConfig(lr=1e-4)
        muon = MuonConfig(lr=1e-4)
        muon_adamw = MuonAdamWConfig(muon=muon, adamw=adamw)
        config = LLMOptimizationConfig(optimizer_config=muon_adamw)

        self.assertIs(config.optimizer_config, muon_adamw)

    def test_llm_module_exports_short_names(self):
        model = TinyLLM()
        config = llm_optim.OptimizationConfig.from_preset("sft")
        optimizer = llm_optim.create_optimizer(model, config)

        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_create_llm_lightning_optimizers_returns_lightning_style_dict(self):
        model = TinyLLM()
        configured: LLMLightningOptimizerConfig = create_llm_lightning_optimizers(
            model,
            LLMOptimizationConfig.from_preset(
                "sft",
                scheduler=[("linear", 2), ("cosine", 8)],
            ),
        )

        self.assertEqual(set(configured), {"optimizer", "lr_scheduler"})
        self.assertIsInstance(configured["optimizer"], torch.optim.AdamW)
        self.assertEqual(configured["lr_scheduler"]["interval"], "step")

    def test_composite_optimizer_scheduler_state_round_trip(self):
        model = TinyLLM()
        optimizer = create_muon_adamw_optimizer(
            model,
            muon=MuonConfig(lr=1.0),
            adamw=AdamWConfig(lr=1.0),
            excluded_modules=(model.lm_head,),
        )
        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(
                phases=(
                    SchedulerPhaseConfig(
                        shape=CurveShape.LINEAR,
                        duration_steps=2,
                        start_lr_ratio=0.0,
                        end_lr_ratio=1.0,
                    ),
                    SchedulerPhaseConfig(
                        shape=CurveShape.COSINE,
                        duration_steps=4,
                        end_lr_ratio=0.1,
                    ),
                )
            ),
        )

        for _ in range(3):
            optimizer.step()
            scheduler.step()

        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()
        expected_lrs = [group["lr"] for group in optimizer.param_groups]

        restored_model = TinyLLM()
        restored_optimizer = create_muon_adamw_optimizer(
            restored_model,
            muon=MuonConfig(lr=1.0),
            adamw=AdamWConfig(lr=1.0),
            excluded_modules=(restored_model.lm_head,),
        )
        restored_scheduler = create_scheduler(
            restored_optimizer,
            SchedulerConfig(
                phases=(
                    SchedulerPhaseConfig(
                        shape=CurveShape.LINEAR,
                        duration_steps=2,
                        start_lr_ratio=0.0,
                        end_lr_ratio=1.0,
                    ),
                    SchedulerPhaseConfig(
                        shape=CurveShape.COSINE,
                        duration_steps=4,
                        end_lr_ratio=0.1,
                    ),
                )
            ),
        )

        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)

        self.assertEqual(restored_scheduler.last_epoch, scheduler.last_epoch)
        self.assertEqual([group["lr"] for group in restored_optimizer.param_groups], expected_lrs)

        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()

        self.assertEqual(
            [group["lr"] for group in restored_optimizer.param_groups],
            [group["lr"] for group in optimizer.param_groups],
        )

    def test_llm_lightning_optimizers_resume_from_checkpoint(self):
        class Module(LightningModule):
            def __init__(self):
                super().__init__()
                self.model = nn.Linear(2, 1)
                self.optim_config = LLMOptimizationConfig.from_preset(
                    "sft",
                    scheduler=[("linear", 1), ("cosine", 2)],
                )

            def training_step(self, batch, batch_idx):
                inputs, targets = batch
                predictions = self.model(inputs)
                return torch.nn.functional.mse_loss(predictions, targets)

            def configure_optimizers(self):
                return create_llm_lightning_optimizers(self.model, self.optim_config)

        dataset = TensorDataset(torch.ones(4, 2), torch.zeros(4, 1))
        dataloader = DataLoader(dataset, batch_size=2)

        with TemporaryDirectory() as tmp_dir:
            trainer = Trainer(
                default_root_dir=tmp_dir,
                max_steps=2,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                accelerator="cpu",
                devices=1,
            )
            trainer.fit(Module(), train_dataloaders=dataloader)
            checkpoint_path = f"{tmp_dir}/resume.ckpt"
            trainer.save_checkpoint(checkpoint_path)

            resumed_trainer = Trainer(
                default_root_dir=tmp_dir,
                max_steps=3,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
                accelerator="cpu",
                devices=1,
            )
            resumed_trainer.fit(Module(), train_dataloaders=dataloader, ckpt_path=checkpoint_path)

            self.assertEqual(resumed_trainer.global_step, 3)

    def test_llm_config_from_preset_rejects_invalid_scheduler_input(self):
        with self.assertRaisesRegex(TypeError, "scheduler"):
            LLMOptimizationConfig.from_preset("sft", scheduler="linear")

    def test_create_scheduler_warmup_and_decay(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(
                phases=(
                    SchedulerPhaseConfig(
                        shape="linear",
                        duration_steps=2,
                        start_lr_ratio=0.0,
                        end_lr_ratio=1.0,
                    ),
                    SchedulerPhaseConfig(
                        shape="linear",
                        duration_steps=8,
                        end_lr_ratio=0.1,
                    ),
                )
            ),
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        for _ in range(20):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_create_scheduler_supports_wsd_composition(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(
                phases=(
                    SchedulerPhaseConfig(
                        shape="linear",
                        duration_steps=2,
                        start_lr_ratio=0.0,
                        end_lr_ratio=1.0,
                    ),
                    SchedulerPhaseConfig(
                        shape="constant",
                        duration_steps=3,
                        end_lr_ratio=1.0,
                    ),
                    SchedulerPhaseConfig(
                        shape="cosine",
                        duration_steps=5,
                        end_lr_ratio=0.1,
                    ),
                )
            ),
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.5)
        optimizer.step()
        scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)
        for _ in range(10):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.1)

    def test_create_scheduler_supports_infinite_constant_tail(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        scheduler = create_scheduler(
            optimizer,
            SchedulerConfig(
                phases=(
                    SchedulerPhaseConfig(
                        shape="linear",
                        duration_steps=2,
                        start_lr_ratio=0.0,
                        end_lr_ratio=1.0,
                    ),
                    SchedulerPhaseConfig(shape="constant", duration_steps=-1),
                )
            ),
        )

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        for _ in range(20):
            optimizer.step()
            scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0)

    def test_scheduler_phase_config_accepts_schedule_shape_enum(self):
        config = SchedulerPhaseConfig(
            shape=CurveShape.LINEAR,
            duration_steps=1,
            start_lr_ratio=0.0,
            end_lr_ratio=1.0,
        )

        self.assertIs(config.shape, CurveShape.LINEAR)

    def test_make_scheduler_config_helper_accepts_shape_and_duration_steps(self):
        config = make_scheduler_config(
            ("linear", 2),
            ("constant", 3),
            ("cosine", 5),
        )

        self.assertEqual(
            [phase.shape for phase in config.phases],
            [CurveShape.LINEAR, CurveShape.CONSTANT, CurveShape.COSINE],
        )
        self.assertEqual([phase.duration_steps for phase in config.phases], [2, 3, 5])
        self.assertEqual(config.phases[0].start_lr_ratio, 0.0)
        self.assertEqual(config.phases[0].end_lr_ratio, 1.0)
        self.assertEqual(config.phases[1].end_lr_ratio, 1.0)
        self.assertEqual(config.phases[2].end_lr_ratio, 0.1)

    def test_make_scheduler_config_rejects_non_final_infinite_phase(self):
        with self.assertRaisesRegex(ValueError, "unreachable"):
            SchedulerConfig(
                phases=(
                    SchedulerPhaseConfig(shape="constant", duration_steps=-1),
                    SchedulerPhaseConfig(shape="cosine", duration_steps=-1),
                )
            )

    def test_scheduler_config_rejects_infinite_non_constant_phase(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            SchedulerConfig(
                phases=(
                    SchedulerPhaseConfig(shape="cosine", duration_steps=-1),
                )
            )

    def test_llm_muon_defaults_reuse_adamw_lr_and_weight_decay(self):
        model = TinyLLM()
        optimizer = create_llm_optimizer(
            model,
            LLMOptimizationConfig.from_preset("pretrain", optimizer="muon"),
        )

        self.assertIsInstance(optimizer, CompositeOptimizer)
        muon_group = optimizer.optimizers["muon"].param_groups[0]
        self.assertEqual(muon_group["lr"], 3e-4)
        self.assertEqual(muon_group["weight_decay"], 0.1)
        self.assertEqual(muon_group["adjust_lr_fn"], MuonAdjustLRFn.MATCH_RMS_ADAMW)


class OptimConfigTest(unittest.TestCase):
    def test_adamw_config_rejects_invalid_betas(self):
        with self.assertRaisesRegex(ValueError, "betas"):
            AdamWConfig(lr=1e-4, betas=(0.9, 1.0))

    def test_muon_config_defaults_to_match_rms_adamw(self):
        self.assertEqual(MuonConfig(lr=1e-4).adjust_lr_fn, MuonAdjustLRFn.MATCH_RMS_ADAMW)

    def test_muon_config_accepts_adjust_lr_fn_string(self):
        config = MuonConfig(lr=1e-4, adjust_lr_fn="original")

        self.assertEqual(config.adjust_lr_fn, MuonAdjustLRFn.ORIGINAL)

    def test_muon_config_rejects_invalid_adjust_lr_fn(self):
        with self.assertRaisesRegex(ValueError, "adjust_lr_fn"):
            MuonConfig(lr=1e-4, adjust_lr_fn="scaled")


if __name__ == "__main__":
    unittest.main()
