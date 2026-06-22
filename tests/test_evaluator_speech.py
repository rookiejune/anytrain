import unittest

import torch

from anytrain.evaluator.speech import UTMOSEvaluator, WhisperASREvaluator


class FakeWhisperBackend:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def transcribe(self, audio, sample_rate, **decode_options):
        self.calls.append((audio, sample_rate, decode_options))
        return self.output


class FakeTextEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, prediction_text, reference_text):
        self.calls.append((prediction_text, reference_text))
        return {"bleu": 99.0, "wer": 0.25, "chrf": 88.0}


class FakeUTMOSBackend:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def score(self, audio, sample_rate):
        self.calls.append((audio, sample_rate))
        return self.output


class SpeechEvaluatorTest(unittest.TestCase):
    def test_whisper_asr_calls_backend_and_returns_text_metrics(self):
        audio = torch.zeros(2, 16000)
        backend = FakeWhisperBackend(["hello world", "good morning"])
        text_evaluator = FakeTextEvaluator()
        evaluator = WhisperASREvaluator(
            backend=backend,
            text_evaluator=text_evaluator,
            decode_options={"language": "en"},
        )

        metrics = evaluator(
            audio,
            16000,
            reference_text=["hello world", "good morning"],
            temperature=0.0,
        )

        self.assertEqual(set(metrics), {"bleu", "wer", "chrf"})
        self.assertEqual(metrics["bleu"], 99.0)
        self.assertEqual(metrics["wer"], 0.25)
        self.assertEqual(metrics["chrf"], 88.0)
        self.assertEqual(len(backend.calls), 1)
        self.assertIs(backend.calls[0][0], audio)
        self.assertEqual(backend.calls[0][1], 16000)
        self.assertEqual(backend.calls[0][2], {"language": "en", "temperature": 0.0})
        self.assertEqual(
            text_evaluator.calls,
            [(["hello world", "good morning"], ["hello world", "good morning"])],
        )

    def test_whisper_asr_requires_reference_text(self):
        backend = FakeWhisperBackend("hello world")
        evaluator = WhisperASREvaluator(backend=backend, text_evaluator=FakeTextEvaluator())

        with self.assertRaisesRegex(ValueError, "reference_text is required"):
            evaluator(torch.zeros(16000), 16000)

        self.assertEqual(backend.calls, [])

    def test_whisper_asr_rejects_prediction_reference_count_mismatch(self):
        evaluator = WhisperASREvaluator(
            backend=FakeWhisperBackend("hello world"),
            text_evaluator=FakeTextEvaluator(),
        )

        with self.assertRaisesRegex(ValueError, "counts must match"):
            evaluator(torch.zeros(16000), 16000, reference_text=["hello", "world"])

    def test_whisper_asr_requires_backend(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit backend"):
            WhisperASREvaluator(text_evaluator=FakeTextEvaluator())

    def test_utmos_returns_batch_mean(self):
        audio = torch.zeros(3, 16000)
        backend = FakeUTMOSBackend([3.0, 4.0, 5.0])
        evaluator = UTMOSEvaluator(backend=backend)

        metrics = evaluator(audio, 16000)

        self.assertEqual(metrics, {"utmos": 4.0})
        self.assertEqual(backend.calls, [(audio, 16000)])

    def test_utmos_accepts_integer_scores_from_backend(self):
        evaluator = UTMOSEvaluator(backend=FakeUTMOSBackend([3, 5]))

        metrics = evaluator(torch.zeros(2, 16000), 16000)

        self.assertEqual(metrics, {"utmos": 4.0})

    def test_utmos_rejects_boolean_scores(self):
        evaluator = UTMOSEvaluator(backend=FakeUTMOSBackend([True]))

        with self.assertRaisesRegex(TypeError, r"score\[0\]"):
            evaluator(torch.zeros(16000), 16000)

    def test_utmos_requires_backend(self):
        with self.assertRaisesRegex(ValueError, "requires an explicit backend"):
            UTMOSEvaluator()


if __name__ == "__main__":
    unittest.main()
