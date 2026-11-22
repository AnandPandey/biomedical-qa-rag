"""Tests for bmqa.llm.client (FakeLLMClient, AnthropicClient, make_client).

All Anthropic tests use mocks or monkeypatching; no real network or API calls are made.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from bmqa.llm.client import (
    ANSWER_SCHEMA,
    FAKE_REPLY,
    AnthropicClient,
    FakeLLMClient,
    QwenClient,
    make_client,
)
from bmqa.schema import LLMClient, LLMResponse


def test_fake_client_satisfies_protocol():
    client = FakeLLMClient()
    assert isinstance(client, LLMClient)
    assert hasattr(client, "model")
    assert client.model == "fake-llm"


def test_fake_client_deterministic_default_reply():
    client = FakeLLMClient()
    resp = client.generate("Question prompt", max_tokens=100)
    assert isinstance(resp, LLMResponse)
    assert resp.text == FAKE_REPLY
    assert resp.model == "fake-llm"
    assert resp.cached is False
    assert client.last_stop_reason == "end_turn"
    assert client.prompts == ["Question prompt"]


def test_fake_client_callable_reply():
    client = FakeLLMClient(reply=lambda p: f"echo: {p}", model="echo-model")
    resp = client.generate("hello world", max_tokens=50)
    assert resp.text == "echo: hello world"
    assert resp.model == "echo-model"
    assert resp.input_tokens == 2  # len("hello world".split())
    assert resp.output_tokens == 3  # len("echo: hello world".split())


def test_make_client_dry_run_always_returns_fake():
    cfg = {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "effort": "low",
        "max_tokens": 100,
    }
    client = make_client(cfg, run=1, dry_run=True)
    assert isinstance(client, FakeLLMClient)


def test_make_client_unknown_provider_raises_value_error():
    cfg = {"provider": "unsupported_provider", "model": "x", "effort": "low"}
    with pytest.raises(ValueError, match="unknown LLM provider 'unsupported_provider'"):
        make_client(cfg, run=1, dry_run=False)


def test_lazy_import_anthropic_not_imported_at_module_level():
    # Verify that importing bmqa.llm.client does not import the vendor SDK eagerly
    # If anthropic is in sys.modules, delete it temporarily to verify lazy import
    with patch.dict(sys.modules, {"anthropic": None}):
        # Instantiating AnthropicClient when anthropic cannot be imported raises ImportError
        with pytest.raises(ImportError):
            AnthropicClient(model="claude-haiku-5-5", effort="low", run=1)


def test_anthropic_client_mocked_request_and_cache(tmp_path):
    mock_anthropic = MagicMock()
    mock_sdk_instance = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_sdk_instance

    # Mock response object from SDK
    mock_text_block = MagicMock()
    mock_text_block.type = "text"
    mock_text_block.text = '{"answer": "yes", "rationale": "mocked", "citations": []}'

    mock_response = MagicMock()
    mock_response.content = [mock_text_block]
    mock_response.usage.input_tokens = 45
    mock_response.usage.output_tokens = 18
    mock_response.stop_reason = "end_turn"

    mock_sdk_instance.messages.create.return_value = mock_response

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        client = AnthropicClient(model="claude-haiku-5-5", effort="low", run=1, cache_dir=tmp_path)
        assert isinstance(client, LLMClient)

        # 1. First call -> Cache miss -> calls mock API
        resp1 = client.generate("Test prompt 1", max_tokens=1024)
        assert resp1.text == '{"answer": "yes", "rationale": "mocked", "citations": []}'
        assert resp1.input_tokens == 45
        assert resp1.output_tokens == 18
        assert resp1.cached is False
        assert client.last_stop_reason == "end_turn"

        # Verify correct parameters were passed to messages.create
        expected_output_config = {
            "effort": "low",
            "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
        }
        mock_sdk_instance.messages.create.assert_called_once_with(
            model="claude-haiku-5-5",
            max_tokens=1024,
            output_config=expected_output_config,
            messages=[{"role": "user", "content": "Test prompt 1"}],
        )

        # 2. Second call with same prompt -> Cache hit -> Does NOT call mock API again
        resp2 = client.generate("Test prompt 1", max_tokens=1024)
        assert resp2.text == resp1.text
        assert resp2.cached is True
        assert mock_sdk_instance.messages.create.call_count == 1  # No second API call!


def test_anthropic_client_error_propagation(tmp_path):
    mock_anthropic = MagicMock()
    mock_sdk_instance = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_sdk_instance
    mock_sdk_instance.messages.create.side_effect = RuntimeError(
        "API authentication failed (mocked)"
    )

    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        client = AnthropicClient(model="claude-haiku-5-5", effort="low", run=1, cache_dir=tmp_path)
        with pytest.raises(RuntimeError, match="API authentication failed"):
            client.generate("Test prompt", max_tokens=100)


def test_lazy_import_qwen_not_imported_at_module_level():
    with patch.dict(sys.modules, {"torch": None, "transformers": None}):
        with pytest.raises(ImportError):
            QwenClient(model="Qwen/Qwen2.5-7B-Instruct", run=1)


def test_qwen_client_mocked_generate_and_cache(tmp_path):
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()

    mock_transformers = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    mock_tokenizer.chat_template = "{{ messages }}"
    mock_tokenizer.apply_chat_template.return_value = "<user> Test prompt </user>"
    mock_tokenizer.eos_token_id = 999

    # Mock tokenized inputs
    mock_tensor = MagicMock()
    mock_tensor.shape = (1, 10)  # 10 input tokens
    mock_inputs = {"input_ids": mock_tensor}
    mock_tensor.to.return_value = mock_tensor
    mock_tokenizer.return_value = mock_inputs

    # Mock generated output: 10 input tokens + 5 generated tokens = 15 tokens
    mock_outputs = MagicMock()
    mock_generated_ids = [101, 102, 103, 104, 105]
    mock_outputs.__getitem__.return_value = [0] * 10 + mock_generated_ids
    mock_model.generate.return_value = mock_outputs

    mock_tokenizer.decode.return_value = (
        '{"answer": "no", "rationale": "mocked qwen", "citations": []}'
    )

    with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
        client = QwenClient(
            model="Qwen/Qwen2.5-7B-Instruct", run=1, cache_dir=tmp_path, load_in_4bit=False
        )
        assert isinstance(client, LLMClient)

        # 1. First call -> Cache miss -> runs inference
        resp1 = client.generate("Test prompt", max_tokens=256)
        assert resp1.text == '{"answer": "no", "rationale": "mocked qwen", "citations": []}'
        assert resp1.cached is False
        assert client.last_stop_reason == "end_turn"
        assert mock_model.generate.call_count == 1

        # 2. Second call -> Cache hit -> Does NOT call generate again
        resp2 = client.generate("Test prompt", max_tokens=256)
        assert resp2.text == resp1.text
        assert resp2.cached is True
        assert mock_model.generate.call_count == 1


def test_make_client_qwen():
    cfg = {
        "provider": "qwen",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "max_tokens": 512,
    }
    # Dry run returns FakeLLMClient without importing heavy packages
    client = make_client(cfg, run=1, dry_run=True)
    assert isinstance(client, FakeLLMClient)
