import asyncio
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from src import llama_server
from src.llama_server import LlamaServerClient, megakernel_unavailable_reason
from src.runtime_registry import create_runtime_client

ROOT = Path(__file__).resolve().parent.parent
MODEL = "/models/Qwen3.8-27B-Q4_K_M.gguf"


def _load_mk_server():
    path = ROOT / "megakernel" / "server" / "mk_server.py"
    spec = importlib.util.spec_from_file_location("mk_server_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MegakernelSelectionTests(unittest.TestCase):
    def test_ada_mtp_uses_measured_depth(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "8.9")):
            self.assertEqual(llama_server._mtp_spec_args()[-1], "4")

    def test_ampere_keeps_mtp_depth(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.object(llama_server, "_megakernel_gpu_info", return_value=("0", "8.6")):
            self.assertEqual(llama_server._mtp_spec_args()[-1], "2")

    def test_mtp_depth_override_is_checked(self) -> None:
        with mock.patch.dict("os.environ", {"OPENJET_MTP_DRAFT_MAX": "3"}):
            self.assertEqual(llama_server._mtp_spec_args()[-1], "3")
        for invalid in ("abc", "0", "9"):
            with mock.patch.dict("os.environ", {"OPENJET_MTP_DRAFT_MAX": invalid}):
                with self.assertRaises(ValueError):
                    llama_server._mtp_spec_args()

    def test_other_models_are_not_served_by_the_megakernel(self) -> None:
        for model in ("/m/Qwen3.6-27B-UD-IQ3_XXS.gguf", "/m/Qwen3.8-27B-Q6_K.gguf", "/m/qwen3.8-27b-q4_k_m.bin"):
            self.assertIn("only supports", megakernel_unavailable_reason(model))

    def test_missing_build_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=None):
            self.assertIn("not built", megakernel_unavailable_reason(MODEL))

    def test_unsupported_gpu_is_reported(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu", return_value=None):
            self.assertIn("RTX 30/40/50", megakernel_unavailable_reason(MODEL))

    def test_supported_setup_has_no_reason(self) -> None:
        with mock.patch.object(llama_server, "_megakernel_dir", return_value=ROOT), \
                mock.patch.object(llama_server, "_megakernel_gpu", return_value="0"):
            self.assertIsNone(megakernel_unavailable_reason(MODEL))

    def test_backend_setting_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            LlamaServerClient(model=MODEL, backend="vllm")

    def test_registry_passes_the_backend_setting(self) -> None:
        cfg = {"runtime": "llama_cpp", "llama_model": MODEL, "llama_backend": "llama"}
        client = create_runtime_client(cfg)
        self.assertEqual(client.backend, "llama")
        self.assertEqual(create_runtime_client({"runtime": "llama_cpp", "llama_model": MODEL}).backend, "auto")


def _nvidia_smi(*lines: str):
    return mock.patch.object(
        llama_server.subprocess, "run",
        return_value=subprocess.CompletedProcess([], 0, stdout="\n".join(lines) + "\n", stderr=""),
    )


class MegakernelGpuDetectionTests(unittest.TestCase):
    def test_a_qualifying_gpu_after_the_first_is_found(self) -> None:
        with _nvidia_smi("0, GPU-aaa, 7.5, 8192", "1, GPU-bbb, 8.6, 24576"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertEqual(llama_server._megakernel_gpu(), "1")

    def test_cuda_visible_devices_limits_the_choice(self) -> None:
        lines = ("0, GPU-aaa, 8.6, 24576", "1, GPU-bbb, 7.5, 8192")
        with _nvidia_smi(*lines), mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertIsNone(llama_server._megakernel_gpu())
        with _nvidia_smi(*lines), mock.patch.dict(llama_server.os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-aaa,1"}):
            self.assertEqual(llama_server._megakernel_gpu(), "0")

    def test_small_or_other_arch_gpus_do_not_qualify(self) -> None:
        with _nvidia_smi("0, GPU-aaa, 8.6, 12288", "1, GPU-bbb, 7.5, 24576", "2, GPU-ccc, 9.0, 81559"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertIsNone(llama_server._megakernel_gpu())

    def test_ada_16gb_gpu_qualifies(self) -> None:
        with _nvidia_smi("0, GPU-aaa, 8.9, 16380"), \
                mock.patch.dict(llama_server.os.environ, {}, clear=True):
            self.assertEqual(llama_server._megakernel_gpu(), "0")

    def test_ada_and_blackwell_24gb_gpus_qualify(self) -> None:
        for cap, mem in (("8.9", "24564"), ("12.0", "32607")):
            with _nvidia_smi(f"0, GPU-aaa, {cap}, {mem}"), \
                    mock.patch.dict(llama_server.os.environ, {}, clear=True):
                self.assertEqual(llama_server._megakernel_gpu(), "0")


class MegakernelStartTests(unittest.TestCase):
    def _client(self, backend: str) -> LlamaServerClient:
        client = LlamaServerClient(model=MODEL, binary="/opt/llama-server", backend=backend)
        client._cleanup_stale_inference_processes = mock.AsyncMock()
        client._resolve_device = mock.Mock(return_value="vulkan")
        return client

    def _run(self, coro):
        return asyncio.run(coro)

    def test_required_megakernel_fails_on_a_non_cuda_device(self) -> None:
        client = self._client("megakernel")
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            with self.assertRaisesRegex(RuntimeError, "desktop CUDA GPU"):
                self._run(client.start())
        client._start_once.assert_not_called()

    def test_required_megakernel_fails_on_jetson(self) -> None:
        client = self._client("megakernel")
        client._resolve_device.return_value = "cuda"
        client._is_jetson_platform = mock.Mock(return_value=True)
        client._ensure_jetson_clocks_sudoers = mock.Mock()
        client._maximize_gpu_clocks = mock.Mock()
        client._prepare_memory_for_launch = mock.AsyncMock(return_value=None)
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            with self.assertRaisesRegex(RuntimeError, "jetson"):
                self._run(client.start())
        client._start_once.assert_not_called()

    def test_fallback_to_llama_server_resets_the_active_backend(self) -> None:
        client = self._client("auto")
        client.active_backend = "megakernel"
        client._start_once = mock.AsyncMock()
        with mock.patch.object(llama_server, "assert_endpoint_allowed"):
            self._run(client.start())
        client._start_once.assert_called_once()
        self.assertEqual(client.active_backend, "llama.cpp")

    def test_stopping_the_server_stops_the_stderr_reader(self) -> None:
        async def scenario() -> None:
            client = LlamaServerClient(model=MODEL)
            stderr = asyncio.StreamReader()
            client._proc = mock.Mock(returncode=0, stderr=stderr)
            client._stderr_task = asyncio.create_task(client._capture_startup_stderr(llama_server.deque()))
            await asyncio.sleep(0)
            task = client._stderr_task
            await client._stop_server()
            self.assertIsNone(client._proc)
            stderr.feed_data(b"late line\n")
            with self.assertRaises(asyncio.CancelledError):
                await task
            await client._http.aclose()

        self._run(scenario())


class MegakernelServerParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mk = _load_mk_server()

    def _feed_all(self, parser, chunks):
        out = []
        for c in chunks:
            out += parser.feed(c)
        out += parser.feed("", final=True)
        merged = {"reasoning": "", "content": ""}
        for kind, text in out:
            merged[kind] += text
        return merged

    def test_reasoning_and_content_are_split_across_chunk_boundaries(self) -> None:
        parser = self.mk.OutputParser(thinking=True, tools=[])
        text = "Think about it.\n</think>\n\nThe answer is 4."
        merged = self._feed_all(parser, [text[i:i + 3] for i in range(0, len(text), 3)])
        self.assertEqual(merged["reasoning"], "Think about it.\n")
        self.assertEqual(merged["content"], "The answer is 4.")

    def test_tool_call_markup_is_held_back_and_parsed(self) -> None:
        tools = [{"type": "function", "function": {"name": "read_file", "parameters": {
            "type": "object", "properties": {"path": {"type": "string"}, "max_lines": {"type": "integer"}}}}}]
        parser = self.mk.OutputParser(thinking=False, tools=tools)
        text = ("Let me look.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\n/etc/hostname\n</parameter>\n"
                "<parameter=max_lines>\n10\n</parameter>\n</function>\n</tool_call>")
        merged = self._feed_all(parser, [text[i:i + 5] for i in range(0, len(text), 5)])
        self.assertEqual(merged["content"], "Let me look.")
        calls = self.mk.parse_tool_calls(parser.tool_text, tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "/etc/hostname", "max_lines": 10})

    def test_string_parameters_are_not_json_decoded(self) -> None:
        tools = [{"type": "function", "function": {"name": "run", "parameters": {
            "type": "object", "properties": {"cmd": {"type": "string"}}}}}]
        calls = self.mk.parse_tool_calls("<tool_call>\n<function=run>\n<parameter=cmd>\n123\n</parameter>\n</function>\n</tool_call>", tools)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"cmd": "123"})

    def test_assistant_tool_call_arguments_are_decoded_for_the_template(self) -> None:
        msgs = self.mk.normalize_messages([{"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]}])
        self.assertEqual(msgs[0]["content"], "")
        self.assertEqual(msgs[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})

    def test_images_are_rejected(self) -> None:
        with self.assertRaises(self.mk.BadRequest):
            self.mk.normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])


class MegakernelReasoningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mk = _load_mk_server()

    def server(self):
        server = self.mk.Server.__new__(self.mk.Server)
        server.template = self.mk.make_template(
            "effort={{ reasoning_effort|default('xhigh') }};"
            "preserve={{ preserve_thinking|default(false)|tojson }}\n"
            "{% if enable_thinking is defined and enable_thinking is false %}"
            "{{ '<think>\\n\\n</think>\\n\\n' }}"
            "{% else %}{{ '<think>\\n' }}{% endif %}"
        )
        return server

    def render(self, extra):
        body = {"messages": [{"role": "user", "content": "Quanto é 2 + 2?"}], **extra}
        with mock.patch.object(self.mk, "log"):
            return self.server().render(body)

    def test_top_level_efforts_reach_the_native_template(self):
        for wire, native in (("low", "low"), ("medium", "medium"), ("high", "xhigh"), ("xhigh", "xhigh")):
            with self.subTest(effort=wire):
                text, thinking = self.render({"reasoning_effort": wire})
                self.assertTrue(thinking)
                self.assertIn(f"effort={native};", text)

    def test_custom_provider_thinking_off_disables_reasoning(self):
        for extra in ({"reasoning_effort": "none"}, {"reasoning_effort": "none", "enable_thinking": True}):
            text, thinking = self.render(extra)
            self.assertFalse(thinking)
            self.assertTrue(text.endswith("<think>\n\n</think>\n\n"))

    def test_llama_cpp_provider_split_controls_are_supported(self):
        text, thinking = self.render({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": True}})
        self.assertTrue(thinking)
        self.assertIn("effort=low;", text)

    def test_explicit_disable_wins_over_positive_effort(self):
        for extra in ({"enable_thinking": False, "reasoning_effort": "high"},
                      {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}}):
            self.assertFalse(self.render(extra)[1])

    def test_nested_controls_take_precedence_over_top_level_fields(self):
        text, thinking = self.render({"reasoning_effort": "none", "enable_thinking": False,
                                     "chat_template_kwargs": {"reasoning_effort": "medium", "enable_thinking": True}})
        self.assertTrue(thinking)
        self.assertIn("effort=medium;", text)

    def test_missing_or_null_controls_keep_the_model_default(self):
        for extra in ({}, {"reasoning_effort": None, "enable_thinking": None},
                      {"chat_template_kwargs": None}, {"chat_template_kwargs": {"reasoning_effort": None}}):
            text, thinking = self.render(extra)
            self.assertTrue(thinking)
            self.assertIn("effort=xhigh;", text)
        text, thinking = self.render({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": None}})
        self.assertTrue(thinking)
        self.assertIn("effort=low;", text)

    def test_canonical_and_compatible_requests_render_identically(self):
        for effort in ("low", "medium", "xhigh"):
            self.assertEqual(self.render({"reasoning_effort": effort}),
                             self.render({"chat_template_kwargs": {"reasoning_effort": effort}}))
        self.assertEqual(self.render({"reasoning_effort": "none"}),
                         self.render({"chat_template_kwargs": {"enable_thinking": False}}))

    def test_preserve_thinking_and_input_objects_are_preserved(self):
        body = {"messages": [{"role": "user", "content": "Quanto é 2 + 2?"}],
                "reasoning_effort": "high", "preserve_thinking": True,
                "chat_template_kwargs": {"enable_thinking": True}}
        original = json.loads(json.dumps(body))
        with mock.patch.object(self.mk, "log"):
            text, thinking = self.server().render(body)
        self.assertEqual(body, original)
        self.assertTrue(thinking)
        self.assertIn("preserve=true", text)

    def test_invalid_controls_raise_client_errors(self):
        for body in ({"reasoning_effort": "unsupported"}, {"reasoning_effort": 1},
                     {"reasoning_effort": []}, {"enable_thinking": "false"},
                     {"preserve_thinking": 0}, {"chat_template_kwargs": []},
                     {"chat_template_kwargs": "low"}, {"chat_template_kwargs": {"reasoning_effort": False}}):
            with self.subTest(body=body), self.assertRaises(self.mk.BadRequest):
                self.render(body)

    def test_effective_reasoning_control_is_logged(self):
        with mock.patch.object(self.mk, "log") as log:
            self.server().render({"reasoning_effort": "low"})
            log.assert_called_once_with("reasoning:", "on, effort=low")
        with mock.patch.object(self.mk, "log") as log:
            self.server().render({"reasoning_effort": "none"})
            log.assert_called_once_with("reasoning:", "off")


class MegakernelSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mk = _load_mk_server()

    def server(self):
        server = self.mk.Server.__new__(self.mk.Server)
        server.eng = mock.Mock()
        return server

    def test_defaults_match_gsq_model_sampling(self):
        server = self.server()
        server.sampling({"seed": 42})
        server.eng.set_sampling.assert_called_once_with(1.0, 0.95, 20, 0.0, 42)

    def test_unsupported_sampling_is_rejected_without_silent_changes(self):
        for body in ({"top_k": 40}, {"top_k": 0}, {"presence_penalty": 1},
                     {"frequency_penalty": 0.1}, {"repeat_penalty": 1.1}):
            server = self.server()
            with self.assertRaises(self.mk.BadRequest):
                server.sampling(body)
            server.eng.set_sampling.assert_not_called()

    def test_greedy_ignores_candidate_count(self):
        server = self.server()
        server.sampling({"temperature": 0, "top_k": 0, "seed": 7})
        server.eng.set_sampling.assert_called_once_with(0.0, 1.0, 32, 0.0, 7)


if __name__ == "__main__":
    unittest.main()
