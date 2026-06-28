import os
import sys
import types
import unittest
from unittest.mock import patch

import anytrain.chat as chat_api
from anytrain import __all__ as anytrain_exports
from anytrain.chat import Chat


class ChatTest(unittest.TestCase):
    def test_package_root_does_not_export_chat_client(self):
        self.assertNotIn("Chat", anytrain_exports)

    def test_chat_module_only_exports_chat_client(self):
        self.assertEqual(chat_api.__all__, ["Chat"])

    def test_model_type_uses_lowercase_string_values(self):
        self.assertEqual(chat_api.ModelType.DEEPSEEK.value, "deepseek")
        self.assertEqual(chat_api.ModelType.GLM.value, "glm")

    def test_config_from_env_uses_deepseek_envs(self):
        env = {
            chat_api.DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            chat_api.DEEPSEEK_MODEL_ENV: "deepseek-chat",
            chat_api.DEEPSEEK_API_KEY_ENV: "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = chat_api.config_from_env("deepseek")

        self.assertEqual(config.base_url, "https://example.test/deepseek")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertNotIn("secret", repr(config))

    def test_config_from_env_uses_glm_envs(self):
        env = {
            chat_api.GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            chat_api.GLM_MODEL_ENV: "glm-5.2",
            chat_api.GLM_API_KEY_ENV: "secret",
        }
        with patch.dict(os.environ, env, clear=True):
            config = chat_api.config_from_env(chat_api.ModelType.GLM)

        self.assertEqual(config.base_url, "https://example.test/api/paas/v4")
        self.assertEqual(config.model, "glm-5.2")

    def test_missing_env_fails_explicitly(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, chat_api.DEEPSEEK_BASE_URL_ENV),
        ):
            chat_api.config_from_env(chat_api.ModelType.DEEPSEEK)

    def test_empty_env_fails_explicitly(self):
        env = {
            chat_api.DEEPSEEK_BASE_URL_ENV: "",
            chat_api.DEEPSEEK_MODEL_ENV: "deepseek-chat",
            chat_api.DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            self.assertRaisesRegex(ValueError, chat_api.DEEPSEEK_BASE_URL_ENV),
        ):
            chat_api.config_from_env(chat_api.ModelType.DEEPSEEK)

    def test_unknown_model_type_fails(self):
        with self.assertRaises(ValueError):
            chat_api.config_from_env("unknown")

    def test_deepseek_backend_uses_openai_client(self):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                calls.append({"api_key": api_key, "base_url": base_url})
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            chat_api.DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            chat_api.DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            chat_api.DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            response = Chat("deepseek")("hello")

        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0],
            {
                "api_key": "secret",
                "base_url": "https://example.test/deepseek",
            },
        )
        self.assertEqual(
            calls[1],
            {
                "model": "deepseek-v4-pro",
                "messages": [
                    {"role": "system", "content": chat_api.DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "hello"},
                ],
                "stream": False,
                "reasoning_effort": chat_api.DEEPSEEK_REASONING_EFFORT,
                "extra_body": {"thinking": {"type": "enabled"}},
            },
        )

    def test_deepseek_reuses_openai_client(self):
        client_init_count = 0
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                message = types.SimpleNamespace(content=kwargs["messages"][-1]["content"])
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                nonlocal client_init_count
                client_init_count += 1
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            chat_api.DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            chat_api.DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            chat_api.DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            client = Chat("deepseek")
            self.assertEqual(client("first"), "first")
            self.assertEqual(client("second"), "second")

        self.assertEqual(client_init_count, 1)
        self.assertEqual(
            create_messages,
            [
                [
                    {"role": "system", "content": chat_api.DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                ],
                [
                    {"role": "system", "content": chat_api.DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
            ],
        )

    def test_deepseek_refresh_starts_new_context(self):
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                message = types.SimpleNamespace(content=kwargs["messages"][-1]["content"])
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeOpenAI:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_openai = types.SimpleNamespace(OpenAI=FakeOpenAI)
        env = {
            chat_api.DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            chat_api.DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            chat_api.DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": fake_openai}),
        ):
            client = Chat("deepseek")
            self.assertEqual(client("first"), "first")
            self.assertEqual(client("second", refresh=True), "second")
            client.refresh()
            self.assertEqual(client("third"), "third")

        self.assertEqual(
            create_messages,
            [
                [
                    {"role": "system", "content": chat_api.DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "first"},
                ],
                [
                    {"role": "system", "content": chat_api.DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "second"},
                ],
                [
                    {"role": "system", "content": chat_api.DEEPSEEK_SYSTEM_PROMPT},
                    {"role": "user", "content": "third"},
                ],
            ],
        )

    def test_deepseek_backend_requires_chat_extra(self):
        env = {
            chat_api.DEEPSEEK_BASE_URL_ENV: "https://example.test/deepseek",
            chat_api.DEEPSEEK_MODEL_ENV: "deepseek-v4-pro",
            chat_api.DEEPSEEK_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"openai": None}),
            self.assertRaisesRegex(ImportError, r"anytrain\[chat\]"),
        ):
            Chat("deepseek")("hello")

    def test_glm_backend_uses_zai_client(self):
        calls = []

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                message = types.SimpleNamespace(content="done")
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                calls.append({"api_key": api_key, "base_url": base_url})
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            chat_api.GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            chat_api.GLM_MODEL_ENV: "glm-5.2",
            chat_api.GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            response = Chat("glm")("hello")

        self.assertEqual(response, "done")
        self.assertEqual(
            calls[0],
            {
                "api_key": "secret",
                "base_url": "https://example.test/api/paas/v4",
            },
        )
        self.assertEqual(
            calls[1],
            {
                "model": "glm-5.2",
                "messages": [{"role": "user", "content": "hello"}],
                "thinking": {"type": "enabled"},
                "max_tokens": chat_api.GLM_MAX_TOKENS,
                "temperature": chat_api.GLM_TEMPERATURE,
            },
        )

    def test_glm_reuses_context_until_refresh(self):
        create_messages = []

        class FakeCompletions:
            def create(self, **kwargs):
                create_messages.append(kwargs["messages"])
                message = types.SimpleNamespace(content=kwargs["messages"][-1]["content"])
                choice = types.SimpleNamespace(message=message)
                return types.SimpleNamespace(choices=[choice])

        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=FakeCompletions(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            chat_api.GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            chat_api.GLM_MODEL_ENV: "glm-5.2",
            chat_api.GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            client = Chat("glm")
            self.assertEqual(client("first"), "first")
            self.assertEqual(client("second"), "second")
            self.assertEqual(client("third", refresh=True), "third")

        self.assertEqual(
            create_messages,
            [
                [{"role": "user", "content": "first"}],
                [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "first"},
                    {"role": "user", "content": "second"},
                ],
                [{"role": "user", "content": "third"}],
            ],
        )

    def test_glm_backend_requires_chat_extra(self):
        env = {
            chat_api.GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            chat_api.GLM_MODEL_ENV: "glm-5.2",
            chat_api.GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": None}),
            self.assertRaisesRegex(ImportError, r"anytrain\[chat\]"),
        ):
            Chat("glm")("hello")

    def test_chat_rejects_empty_prompt_before_backend(self):
        class FakeZhipuAiClient:
            def __init__(self, *, api_key, base_url):
                self.chat = types.SimpleNamespace(
                    completions=types.SimpleNamespace(),
                )

        fake_zai = types.SimpleNamespace(ZhipuAiClient=FakeZhipuAiClient)
        env = {
            chat_api.GLM_BASE_URL_ENV: "https://example.test/api/paas/v4",
            chat_api.GLM_MODEL_ENV: "glm-5.2",
            chat_api.GLM_API_KEY_ENV: "secret",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.dict(sys.modules, {"zai": fake_zai}),
        ):
            client = Chat("glm")
            with self.assertRaisesRegex(ValueError, "prompt"):
                client("")


if __name__ == "__main__":
    unittest.main()
