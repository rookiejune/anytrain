import sys
import unittest
from unittest.mock import patch

import torch

from anytrain.tts import TTSOptions, TTSOutput, TTSTokens
from anytrain.tts.moss import MossTTS, MossTTSConfig


class TTSProtocolTest(unittest.TestCase):
    def test_output_requires_channel_time_waveform(self):
        with self.assertRaisesRegex(ValueError, "channels, time"):
            TTSOutput(
                waveform=torch.zeros(1, 1, 8),
                sample_rate=24000,
                duration=1.0,
            )

    def test_options_validate_sampling_parameters(self):
        with self.assertRaisesRegex(ValueError, "top_p"):
            TTSOptions(top_p=1.2)

        with self.assertRaisesRegex(ValueError, "temperature"):
            TTSOptions(temperature=0.0)

    def test_tokens_build_model_inputs(self):
        tokens = TTSTokens(
            input_ids=torch.tensor([[1, 2]]),
            attention_mask=torch.tensor([[1, 1]]),
            extra_tensors={"prompt_ids": torch.tensor([[3]])},
        )

        self.assertEqual(set(tokens.model_inputs()), {"attention_mask", "input_ids", "prompt_ids"})


class MossTTSImportTest(unittest.TestCase):
    def test_import_tts_does_not_import_transformers_dependency(self):
        sys.modules.pop("transformers", None)

        import anytrain.tts

        self.assertIn("TTSOutput", anytrain.tts.__all__)
        self.assertNotIn("transformers", sys.modules)

    def test_from_pretrained_requires_optional_dependency_when_missing(self):
        with patch(
            "anytrain.tts.moss.tts.load_transformers_auto_model_class",
            side_effect=ImportError("transformers missing"),
        ), self.assertRaisesRegex(ImportError, r"anytrain\[moss-tts\]"):
            MossTTS.from_pretrained(local_files_only=True)


class MossTTSAdapterTest(unittest.TestCase):
    def test_from_pretrained_loads_remote_code_model_without_importing_at_module_import_time(self):
        class FakeAutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                return FakeMossModel(load_args=(model, kwargs))

        with patch(
            "anytrain.tts.moss.tts.load_transformers_auto_model_class",
            return_value=FakeAutoModelForCausalLM,
        ):
            tts = MossTTS.from_pretrained(
                "fake-model",
                cache_dir="storage/moss",
                device="cpu",
                local_files_only=True,
                sample_rate=16000,
                speaker="spk",
                language="zh",
                revision="main",
                runtime_kwargs={"style": "clear"},
                torch_dtype="float32",
            )

        self.assertEqual(tts.config.model, "fake-model")
        self.assertEqual(tts.sample_rate, 16000)
        self.assertEqual(tts.model.load_args[0], "fake-model")
        self.assertEqual(
            tts.model.load_args[1],
            {
                "cache_dir": "storage/moss",
                "local_files_only": True,
                "revision": "main",
                "torch_dtype": "float32",
            },
        )
        self.assertEqual(tts.config.runtime_kwargs, {"style": "clear"})
        self.assertEqual(tts.model_loaded_from, "transformers")

    def test_from_pretrained_sets_transformers_remote_code_kwargs(self):
        class FakeAutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, model, **kwargs):
                return FakeMossModel(load_args=(model, kwargs))

        with patch(
            "anytrain.tts.moss.tts.load_transformers_auto_model_class",
            return_value=FakeAutoModelForCausalLM,
        ):
            tts = MossTTS.from_pretrained(
                "OpenMOSS-Team/MOSS-TTS-Nano",
                device="cpu",
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype="float32",
            )

        self.assertEqual(tts.model_loaded_from, "transformers")
        self.assertEqual(tts.model.load_args[0], "OpenMOSS-Team/MOSS-TTS-Nano")
        self.assertEqual(
            tts.model.load_args[1],
            {
                "local_files_only": True,
                "torch_dtype": "float32",
                "trust_remote_code": True,
            },
        )

    def test_synthesize_runs_tokenize_generate_decode(self):
        model = FakeMossModel()
        tts = MossTTS(
            model=model,
            config=MossTTSConfig(sample_rate=24000, speaker="default", language="zh"),
        )

        output = tts.synthesize(
            "hello",
            TTSOptions(
                speaker="alice",
                max_new_tokens=8,
                temperature=0.7,
                top_p=0.9,
                extra={"style": "clear"},
            ),
        )

        self.assertEqual(tuple(output.waveform.shape), (1, 4))
        self.assertEqual(output.sample_rate, 24000)
        self.assertAlmostEqual(output.duration, 4 / 24000)
        self.assertEqual([call[0] for call in model.calls], ["tokenize", "generate", "decode"])
        self.assertEqual(model.calls[0][2]["speaker"], "alice")
        self.assertEqual(model.calls[0][2]["language"], "zh")
        self.assertEqual(model.calls[0][2]["style"], "clear")
        self.assertEqual(model.calls[1][1]["max_new_tokens"], 8)
        self.assertEqual(model.calls[1][1]["temperature"], 0.7)
        self.assertEqual(model.calls[1][1]["top_p"], 0.9)
        self.assertTrue(torch.equal(model.calls[2][1], torch.tensor([[7, 8]])))

    def test_runtime_kwargs_do_not_leak_load_kwargs(self):
        model = FakeMossModel()
        tts = MossTTS(
            model=model,
            config=MossTTSConfig(
                load_kwargs={"torch_dtype": "float32"},
                runtime_kwargs={"style": "clear"},
            ),
        )

        tts.synthesize("hello", prompt_audio_path="assets/prompt.wav")

        self.assertEqual(model.calls[0][2]["style"], "clear")
        self.assertEqual(model.calls[0][2]["prompt_audio_path"], "assets/prompt.wav")
        self.assertNotIn("torch_dtype", model.calls[0][2])

    def test_explicit_none_option_overrides_config_value(self):
        model = FakeMossModel()
        tts = MossTTS(
            model=model,
            config=MossTTSConfig(speaker="default", language="zh"),
        )

        tts.synthesize("hello", TTSOptions(speaker=None))

        self.assertNotIn("speaker", model.calls[0][2])
        self.assertEqual(model.calls[0][2]["language"], "zh")

    def test_explicit_extra_overrides_config_runtime_kwargs(self):
        model = FakeMossModel()
        tts = MossTTS(
            model=model,
            config=MossTTSConfig(runtime_kwargs={"style": "clear", "speed": 1.0}),
        )

        tts.synthesize("hello", TTSOptions(extra={}), prompt_audio_path="assets/prompt.wav")

        self.assertNotIn("style", model.calls[0][2])
        self.assertNotIn("speed", model.calls[0][2])
        self.assertEqual(model.calls[0][2]["prompt_audio_path"], "assets/prompt.wav")

    def test_synthesize_uses_hf_inference_when_model_exposes_it(self):
        model = FakeInferenceMossModel()
        tts = MossTTS(
            model=model,
            config=MossTTSConfig(model="OpenMOSS-Team/MOSS-TTS-Nano", sample_rate=48000),
            device="cpu",
        )

        output = tts.synthesize(
            "hello",
            TTSOptions(
                max_new_tokens=71,
                temperature=0.8,
                top_p=0.95,
                seed=7,
                extra={
                    "audio_tokenizer_pretrained_name_or_path": "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
                },
            ),
            output_audio_path="outputs/moss.wav",
            prompt_audio_path="assets/audio/zh_1.wav",
        )

        self.assertEqual(tuple(output.waveform.shape), (2, 480))
        self.assertEqual(output.sample_rate, 48000)
        self.assertEqual(output.meta["audio_token_frames"], 71)
        self.assertEqual([call[0] for call in model.calls], ["inference"])
        _, text, output_audio_path, kwargs = model.calls[0]
        self.assertEqual(text, "hello")
        self.assertEqual(str(output_audio_path), "outputs/moss.wav")
        self.assertEqual(kwargs["mode"], "voice_clone")
        self.assertEqual(kwargs["max_new_frames"], 71)
        self.assertEqual(kwargs["do_sample"], True)
        self.assertEqual(kwargs["text_temperature"], 0.8)
        self.assertEqual(kwargs["audio_temperature"], 0.8)
        self.assertEqual(kwargs["text_top_p"], 0.95)
        self.assertEqual(kwargs["audio_top_p"], 0.95)
        self.assertEqual(kwargs["device"], torch.device("cpu"))
        self.assertEqual(kwargs["prompt_audio_path"], "assets/audio/zh_1.wav")

    def test_seeded_inference_restores_global_rng_state(self):
        model = FakeRandomInferenceMossModel()
        tts = MossTTS(
            model=model,
            config=MossTTSConfig(model="OpenMOSS-Team/MOSS-TTS-Nano"),
            device="cpu",
        )
        torch.manual_seed(1234)
        state = torch.random.get_rng_state()

        first = tts.synthesize("hello", TTSOptions(seed=7))
        after_first = torch.random.get_rng_state()
        second = tts.synthesize("hello", TTSOptions(seed=7))

        self.assertTrue(torch.equal(state, after_first))
        self.assertTrue(torch.equal(first.waveform, second.waveform))

    def test_config_hash_is_stable_and_uses_generation_identity(self):
        first = MossTTS(model=FakeMossModel(), config=MossTTSConfig(speaker="a"))
        second = MossTTS(model=FakeMossModel(), config=MossTTSConfig(speaker="a"))
        changed = MossTTS(model=FakeMossModel(), config=MossTTSConfig(speaker="b"))

        self.assertEqual(first.config_hash(), second.config_hash())
        self.assertNotEqual(first.config_hash(), changed.config_hash())


class FakeMossModel:
    def __init__(self, load_args=None):
        self.load_args = load_args
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append(("tokenize", text, kwargs))
        return {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return torch.tensor([[7, 8]])

    def decode(self, generation, **kwargs):
        self.calls.append(("decode", generation, kwargs))
        return {
            "waveform": torch.arange(4, dtype=torch.float32),
            "sample_rate": kwargs["sample_rate"],
            "meta": {"backend": "fake"},
        }


class FakeInferenceMossModel:
    def __init__(self):
        self.calls = []

    def inference(self, text, output_audio_path, **kwargs):
        self.calls.append(("inference", text, output_audio_path, kwargs))
        return {
            "audio_path": str(output_audio_path),
            "audio_token_ids": torch.zeros(71, 16, dtype=torch.long),
            "sample_rate": 48000,
            "waveform": torch.ones(2, 480),
        }


class FakeRandomInferenceMossModel:
    def inference(self, text, output_audio_path, **kwargs):
        return {"waveform": torch.rand(1, 4)}


if __name__ == "__main__":
    unittest.main()
