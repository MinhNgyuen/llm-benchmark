#!/usr/bin/env python3
"""Regression tests for benchmark.py.

Two real bugs found and confirmed by executing against the actually
installed `ollama` package (0.6.2), not assumed from an older API
shape:

1) get_benchmark_models() indexed each listed model with ["name"], but
   ollama._types.ListResponse.Model has no "name" field at all -- the
   real field is "model". Confirmed via a direct KeyError reproduction
   against the real type before patching.

2) run_benchmark() passed ollama.chat()'s return value (a real
   ollama._types.ChatResponse object) straight into
   OllamaResponse.model_validate(). Pydantic rejects a foreign model
   type directly (confirmed real ValidationError: "Input should be a
   valid dictionary or instance of OllamaResponse") -- .model_dump()
   converts it to a plain dict first, which validates correctly.

Also covers the new --singlemodel argument: bypasses `ollama list`
discovery entirely when given, and takes precedence over
--skip-models.

Usage:
    python3 test_benchmark.py
"""
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ollama._types as ollama_types  # noqa: E402
from benchmark import (  # noqa: E402
    OllamaResponse,
    get_benchmark_models,
    run_benchmark,
    main,
)


def _make_real_chat_response(**overrides):
    """Builds a genuine ollama._types.ChatResponse, not a dict stand-in
    -- this is what run_benchmark() actually receives from a real
    ollama.chat() call, and is exactly the shape that previously broke
    model_validate()."""
    fields = dict(
        model="llama2:13b",
        created_at="2026-07-28T18:06:27Z",
        message=ollama_types.Message(role="assistant", content="hello"),
        done=True,
        total_duration=1_000_000_000,
        load_duration=100_000_000,
        prompt_eval_count=10,
        prompt_eval_duration=50_000_000,
        eval_count=20,
        eval_duration=200_000_000,
    )
    fields.update(overrides)
    return ollama_types.ChatResponse(**fields)


class GetBenchmarkModelsFieldNameTests(unittest.TestCase):
    def test_real_model_field_name_is_model_not_name(self):
        # Direct confirmation of the real type's actual field, so this
        # test fails loudly if a future ollama release renames it
        # again, rather than silently drifting.
        m = ollama_types.ListResponse.Model(model="llama2:13b")
        self.assertEqual(m["model"], "llama2:13b")
        with self.assertRaises(KeyError):
            _ = m["name"]

    def test_get_benchmark_models_extracts_real_field(self):
        fake_list_response = ollama_types.ListResponse(
            models=[
                ollama_types.ListResponse.Model(model="llama2:13b"),
                ollama_types.ListResponse.Model(model="llama2:latest"),
            ]
        )
        with patch("benchmark.ollama.list", return_value=fake_list_response):
            names = get_benchmark_models()
        self.assertEqual(names, ["llama2:13b", "llama2:latest"])

    def test_skip_models_still_works_with_real_field(self):
        fake_list_response = ollama_types.ListResponse(
            models=[
                ollama_types.ListResponse.Model(model="llama2:13b"),
                ollama_types.ListResponse.Model(model="llama2:latest"),
            ]
        )
        with patch("benchmark.ollama.list", return_value=fake_list_response):
            names = get_benchmark_models(skip_models=["llama2:13b"])
        self.assertEqual(names, ["llama2:latest"])


class RunBenchmarkResponseValidationTests(unittest.TestCase):
    def test_non_verbose_real_chat_response_validates_correctly(self):
        real_response = _make_real_chat_response()
        with patch("benchmark.ollama.chat", return_value=real_response):
            result = run_benchmark("llama2:13b", "why is the sky blue?", verbose=False)
        self.assertIsInstance(result, OllamaResponse)
        self.assertEqual(result.model, "llama2:13b")
        self.assertEqual(result.prompt_eval_count, 10)
        self.assertEqual(result.eval_count, 20)

    def test_verbose_streaming_real_chat_response_validates_correctly(self):
        chunk1 = _make_real_chat_response(done=False)
        chunk2 = _make_real_chat_response(done=True)
        with patch("benchmark.ollama.chat", return_value=iter([chunk1, chunk2])):
            result = run_benchmark("llama2:13b", "why is the sky blue?", verbose=True)
        self.assertIsInstance(result, OllamaResponse)
        self.assertTrue(result.done)


class SingleModelArgumentTests(unittest.TestCase):
    def test_singlemodel_bypasses_discovery(self):
        real_response = _make_real_chat_response()
        with patch("sys.argv", ["benchmark.py", "--singlemodel", "llama2:13b", "-p", "hi"]), \
             patch("benchmark.get_benchmark_models") as mock_discover, \
             patch("benchmark.ollama.chat", return_value=real_response):
            main()
        mock_discover.assert_not_called()

    def test_singlemodel_takes_precedence_over_skip_models(self):
        real_response = _make_real_chat_response()
        with patch("sys.argv", ["benchmark.py", "--singlemodel", "llama2:13b",
                                 "--skip-models", "llama2:13b", "-p", "hi"]), \
             patch("benchmark.get_benchmark_models") as mock_discover, \
             patch("benchmark.ollama.chat", return_value=real_response):
            # Should NOT skip llama2:13b just because it's also in
            # --skip-models -- --singlemodel wins.
            main()
        mock_discover.assert_not_called()

    def test_no_singlemodel_still_uses_discovery(self):
        fake_list_response = ollama_types.ListResponse(
            models=[ollama_types.ListResponse.Model(model="llama2:13b")]
        )
        real_response = _make_real_chat_response()
        with patch("sys.argv", ["benchmark.py", "-p", "hi"]), \
             patch("benchmark.ollama.list", return_value=fake_list_response), \
             patch("benchmark.ollama.chat", return_value=real_response):
            main()  # must not raise -- exercises the untouched discovery path


if __name__ == "__main__":
    unittest.main()
