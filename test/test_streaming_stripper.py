# test/test_streaming_stripper.py
"""流式思维链清洗器的单元测试（不依赖 langchain 等重依赖，用 mock 隔离）。"""
import os
import sys
import unittest
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 在导入被测模块前，用假模块顶掉尚未安装的重依赖
for _mod in [
    "dotenv",
    "langchain_core",
    "langchain_core.language_models",
    "langchain_core.language_models.chat_models",
    "langchain_core.messages",
    "langchain_core.output_parsers",
    "langchain_core.prompts",
]:
    sys.modules.setdefault(_mod, MagicMock())

sys.path.insert(0, ".")

from app.services.generation import StreamingReasoningStripper  # noqa: E402


class TestStreamingReasoningStripper(unittest.TestCase):
    def _run(self, chunks):
        s = StreamingReasoningStripper()
        out = "".join(s.feed(c) for c in chunks)
        return out + s.close()

    def test_tag_in_one_chunk(self):
        self.assertEqual(self._run(["你好<think>思考</think>世界"]), "你好世界")

    def test_open_tag_split(self):
        self.assertEqual(self._run(["你好<thi", "nk>思考中</think>世界"]), "你好世界")

    def test_close_tag_split(self):
        self.assertEqual(self._run(["你好<think>思考</thi", "nk>世界"]), "你好世界")

    def test_char_by_char(self):
        self.assertEqual(self._run(list("你好<think>A</think>世界")), "你好世界")

    def test_no_reasoning(self):
        self.assertEqual(self._run(["第一段", "，第二段"]), "第一段，第二段")

    def test_all_reasoning(self):
        self.assertEqual(self._run(["<think>只想不说</think>"]), "")

    def test_unclosed_tag(self):
        self.assertEqual(self._run(["前面<think>想到一半就结束"]), "前面")

    def test_multiple_blocks(self):
        self.assertEqual(self._run(["a<think>x</think>b<think>y</think>c"]), "abc")

    def test_empty_chunks(self):
        self.assertEqual(self._run(["", "正常", ""]), "正常")


if __name__ == "__main__":
    unittest.main(verbosity=2)
