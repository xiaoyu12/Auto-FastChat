"""Regression tests that do not require model weights or llama.cpp."""

import copy
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def llm_utils(monkeypatch):
    config = ModuleType("config")
    config.Config = lambda: SimpleNamespace(smart_llm_model="test-model")
    llama_cpp = ModuleType("llama_cpp")
    model = Mock()
    model.return_value = {"choices": [{"text": "reply"}]}
    llama_cpp.Llama = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "config", config)
    monkeypatch.setitem(sys.modules, "llama_cpp", llama_cpp)
    path = Path(__file__).resolve().parents[1] / "scripts" / "llm_utils.py"
    spec = importlib.util.spec_from_file_location("llm_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inference_receives_full_conversation_in_order(llm_utils):
    messages = [
        {"role": "system", "content": "Follow the task instructions."},
        {"role": "system", "content": "Memory: the report is in reports/."},
        {"role": "user", "content": "Read the report."},
        {"role": "assistant", "content": "I will read reports/result.txt."},
        {"role": "system", "content": "Command result: revenue increased."},
        {"role": "user", "content": "Feedback: include the revenue change."},
        {"role": "user", "content": "Agent follow-up: summarize the findings."},
    ]
    original = copy.deepcopy(messages)

    result = llm_utils.create_chat_completion(
        messages, model="local-model", temperature=0.2, max_tokens=123
    )

    llm_utils.llm.assert_called_once()
    prompt = llm_utils.llm.call_args.args[0]
    positions = [prompt.index(message["content"]) for message in messages]
    assert positions == sorted(positions)
    labels = {"system": "System", "user": "Human", "assistant": "Assistant"}
    for message in messages:
        assert f"### {labels[message['role']]}:\n{message['content']}" in prompt
    assert prompt.endswith("\n\n### Assistant:\n")
    assert llm_utils.llm.call_args.kwargs == {
        "stop": ["Q:", "### Human:"], "echo": False,
        "temperature": 0.2, "max_tokens": 123,
    }
    assert result == "reply"
    assert messages == original


def test_inference_preserves_default_generation_options(llm_utils):
    messages = [{"role": "user", "content": "Hello"}]

    assert llm_utils.create_chat_completion(messages) == "reply"

    assert llm_utils.llm.call_args.kwargs["temperature"] == 0.36
    assert llm_utils.llm.call_args.kwargs["max_tokens"] == 0
