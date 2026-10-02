"""
Chat history in the browser (gr.BrowserState): snapshot and restore.

The snapshot must be plain JSON, must drop the welcome message and the
thinking placeholder, and must stay bounded. Restoring rebuilds both the
display and the LLM context, and tolerates missing or foreign data.
Without BROWSER_STORAGE_SECRET nothing is stored.
"""

import json
import unittest
from unittest.mock import patch

import tests.conftest  # noqa: F401  (httpx/openai stubs)
from src.ui import gradio_app as ga

_ENV = patch.dict("os.environ", {"BROWSER_STORAGE_SECRET": "test-secret"})


def setUpModule():
    _ENV.start()


def tearDownModule():
    _ENV.stop()


def _state(history=None):
    st = ga.AppState()
    st.chat_history = list(history or [])
    return st


class TestSaveChat(unittest.TestCase):
    def test_drops_welcome_and_placeholder(self):
        chatbot = [
            ga._welcome_message(),
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": ga.THINKING_PLACEHOLDER},
            {"role": "assistant", "content": "Answer"},
        ]
        out = ga.save_chat_to_browser(chatbot, _state())
        self.assertEqual(
            out["chatbot"],
            [{"role": "user", "content": "Question"},
             {"role": "assistant", "content": "Answer"}],
        )

    def test_flattens_content_parts_and_is_json(self):
        chatbot = [{"role": "assistant",
                    "content": [{"type": "text", "text": "Teil 1"}, "Teil 2"]}]
        out = ga.save_chat_to_browser(chatbot, _state())
        self.assertEqual(out["chatbot"][0]["content"], "Teil 1\nTeil 2")
        json.dumps(out)  # must not raise

    def test_keeps_llm_history(self):
        hist = [{"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "system", "content": "nicht speichern"}]
        out = ga.save_chat_to_browser([], _state(hist))
        self.assertEqual([m["content"] for m in out["history"]], ["a", "b"])

    def test_bounded(self):
        chatbot = [{"role": "user", "content": f"m{i}"} for i in range(500)]
        hist = [{"role": "user", "content": f"h{i}"} for i in range(500)]
        out = ga.save_chat_to_browser(chatbot, _state(hist))
        self.assertEqual(len(out["chatbot"]), ga.CHAT_STORE_MAX_MESSAGES)
        self.assertEqual(out["chatbot"][-1]["content"], "m499")
        self.assertEqual(len(out["history"]), ga.CHAT_STORE_MAX_HISTORY)


class TestRestoreChat(unittest.TestCase):
    def setUp(self):
        p = patch.object(ga, "_get_ready_state", side_effect=lambda s: s)
        p.start()
        self.addCleanup(p.stop)

    def test_round_trip(self):
        chatbot = [ga._welcome_message(),
                   {"role": "user", "content": "Question"},
                   {"role": "assistant", "content": "Answer"}]
        hist = [{"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"}]
        stored = ga.save_chat_to_browser(chatbot, _state(hist))
        state, restored = ga.restore_chat_from_browser(stored, _state())
        # fresh welcome message first, then the stored conversation
        self.assertTrue(restored[0]["content"].startswith("👋"))
        self.assertEqual(restored[1:], chatbot[1:])
        self.assertEqual(state.chat_history, hist)

    def test_empty_or_foreign_data_gives_welcome_only(self):
        for stored in (None, {}, {"version": 99, "chatbot": [1]}, "kaputt"):
            state, restored = ga.restore_chat_from_browser(stored, _state())
            self.assertEqual(len(restored), 1)
            self.assertEqual(state.chat_history, [])

    def test_skips_malformed_entries(self):
        stored = {"version": 1,
                  "chatbot": [{"role": "user", "content": "ok"},
                              {"role": "system", "content": "x"},
                              {"role": "user", "content": 5}, "text"],
                  "history": [{"role": "user", "content": "ok"}, None]}
        state, restored = ga.restore_chat_from_browser(stored, _state())
        self.assertEqual(restored[1:], [{"role": "user", "content": "ok"}])
        self.assertEqual(state.chat_history, [{"role": "user", "content": "ok"}])


class TestOptIn(unittest.TestCase):
    def test_secret_from_env(self):
        with patch.dict("os.environ", {"BROWSER_STORAGE_SECRET": " abc "}):
            self.assertEqual(ga._browser_storage_secret(), "abc")
            self.assertTrue(ga.browser_storage_enabled())

    def test_off_without_secret(self):
        for value in ("", "   "):
            with self.subTest(value=value), \
                    patch.dict("os.environ", {"BROWSER_STORAGE_SECRET": value}):
                self.assertIsNone(ga._browser_storage_secret())
                self.assertFalse(ga.browser_storage_enabled())

    def test_nothing_saved_without_secret(self):
        chatbot = [{"role": "user", "content": "Question"}]
        with patch.dict("os.environ", {"BROWSER_STORAGE_SECRET": ""}):
            self.assertEqual(ga.save_chat_to_browser(chatbot, _state()), ga.gr.skip())


if __name__ == "__main__":
    unittest.main()
