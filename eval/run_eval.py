#!/usr/bin/env python
# eval/run_eval.py
"""
RAG 质量评估脚本。

两种后端：
  simple - 内置指标，零额外依赖（检索指标 + LLM 判定的忠实度 + Embedding 相关性）
  ragas  - 官方指标（需 pip install -r requirements-eval.txt）

用法：
  python eval/run_eval.py --set eval/golden_set.json
  python eval/run_eval.py --set eval/golden_set.json --backend ragas
  python eval/run_eval.py --set eval/golden_set.json --top-k 8 --output eval/report.json
"""
import argparse
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.core.config import settings                       # noqa: E402
from app.services.generation import NO_ANSWER, strip_reasoning  # noqa: E402
from app.services.rag_engine import get_rag_engine         # noqa: E402

FAITHFULNESS_PROMPT = """请判断【答案】中的每一句陈述，是否都能由【上下文】直接推出（可以合理概括，但不能是上下文没有的信息）。

对每一句输出一行，严格使用格式：序号|是  或  序号|否
不要输出任何解释、标题或多余内容。

【上下文】
{context}

【答案】
{answer}"""

REFUSAL_TOKENS = ("暂无相关信息", "知识库中暂无", NO_ANSWER.strip("。"))


# ---------------- 工具函数 ----------------

def load_cases(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"] if isinstance(data, dict) else data
    cases = [c for c in cases if not c.get("_说明")]
    return cases[:limit] if limit else cases


def split_sentences(text: str) -> List[str]:
    """按中文/英文句末标点切句，过滤过短片段"""
    parts = re.split(r"(?<=[。！？；.!?;])\s*|\n+", text or "")
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def is_refusal(answer: str) -> bool:
    return any(token in answer for token in REFUSAL_TOKENS)


def mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [v for v in values if v is not None]
    return round(sum(valid) / len(valid), 4) if valid else None


def fmt(value: Optional[float]) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


# ---------------- 评估器 ----------------

class Evaluator:
    def __init__(self):
        self.engine = get_rag_engine()

    # ---- 单条用例 ----

    def run_case(self, case: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        question = case["question"]
        history = [tuple(h) for h in case.get("history", [])] or None

        # 检索
        t0 = time.time()
        hits = self.engine.retrieve(question, history, top_k, case.get("source"))
        items = self.engine.expand_to_parents(hits)
        context, sources, references = self.engine.compose(items)
        latency_retrieve = time.time() - t0

        # 生成
        t1 = time.time()
        answer = self.engine.generator.generate(question, context, history)
        latency_generate = time.time() - t1

        recalled = [r["source"] for r in references]
        expected = case.get("reference_sources") or []

        # 检索指标
        hit = 1.0 if any(s in recalled for s in expected) else 0.0 if expected else None
        recall = (
            len([s for s in expected if s in recalled]) / len(expected) if expected else None
        )
        mrr = 0.0
        if expected:
            for rank, src in enumerate(recalled, start=1):
                if src in expected:
                    mrr = 1.0 / rank
                    break

        # 生成指标
        refused = is_refusal(answer)
        faithfulness = None if (refused or not context) else self._faithfulness(context, answer)
        relevancy = self._similarity(question, answer)
        gt = case.get("ground_truth") or ""
        similarity = self._similarity(gt, answer) if gt else None

        # 拒答正确性
        expect_refusal = case.get("expect_refusal") or (not expected)
        refusal_correct = 1.0 if refused == bool(expect_refusal) else 0.0

        return {
            "id": case.get("id"),
            "question": question,
            "answer": answer,
            "contexts": [text for text, _, _ in items],
            "recalled_sources": recalled,
            "expected_sources": expected,
            "ground_truth": gt,
            "hit": hit,
            "recall": recall,
            "mrr": mrr,
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "similarity": similarity,
            "refusal_correct": refusal_correct,
            "refused": refused,
            "latency_retrieve": round(latency_retrieve, 3),
            "latency_generate": round(latency_generate, 3),
        }

    # ---- 内置指标 ----

    def _faithfulness(self, context: str, answer: str) -> Optional[float]:
        """用 LLM 逐句判定答案是否可由上下文推出，返回支持句比例"""
        sentences = split_sentences(answer)
        if not sentences:
            return None

        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
        prompt = FAITHFULNESS_PROMPT.format(context=context[:4000], answer=numbered)
        try:
            raw = self.engine.generator.llm.invoke(prompt)
            text = strip_reasoning(getattr(raw, "content", str(raw)) or "")
        except Exception as exc:
            print(f"  ! 忠实度判定失败: {exc}")
            return None

        total = yes = 0
        for line in text.splitlines():
            m = re.match(r"\s*(\d+)\s*[|｜:：,，、\s]\s*(是|否|yes|no)", line.strip(), re.I)
            if not m:
                continue
            total += 1
            if m.group(2).lower() in ("是", "yes"):
                yes += 1
        return round(yes / total, 4) if total else None

    def _similarity(self, a: str, b: str) -> Optional[float]:
        """用系统自身的 Embedding 计算余弦相似度"""
        if not a.strip() or not b.strip():
            return None
        try:
            va = self.engine.embeddings.embed_query(a)
            vb = self.engine.embeddings.embed_query(b)
        except Exception:
            return None
        return round(cosine(va, vb), 4)


# ---------------- RAGAS 后端 ----------------

def run_ragas(rows: List[Dict[str, Any]], engine) -> Dict[str, Any]:
    """调用 RAGAS 官方指标（需要额外依赖）"""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        sys.exit("未安装 ragas，请先执行: pip install -r requirements-eval.txt")

    # 包装成本项目的 LLM / Embedding，保证与线上配置一致
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper

        llm = LangchainLLMWrapper(engine.generator.llm)
        embeddings = LangchainEmbeddingsWrapper(engine.embeddings)
    except ImportError:
        llm = embeddings = None  # 交由 ragas 使用其默认配置

    usable = [r for r in rows if r["contexts"] and (r["ground_truth"] or True)]
    if not usable:
        return {}

    dataset = Dataset.from_list(
        [
            {
                "question": r["question"],
                "answer": r["answer"],
                "contexts": r["contexts"],
                "ground_truth": r["ground_truth"] or "",
            }
            for r in usable
        ]
    )

    kwargs = {"llm": llm, "embeddings": embeddings} if llm else {}
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        **kwargs,
    )
    return dict(result)


# ---------------- 主流程 ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 质量评估")
    parser.add_argument("--set", required=True, help="黄金测试集 JSON 路径")
    parser.add_argument("--backend", choices=["simple", "ragas"], default="simple")
    parser.add_argument("--top-k", type=int, default=settings.TOP_K)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--output", default=None, help="报告输出路径（JSON）")
    args = parser.parse_args()

    cases = load_cases(args.set, args.limit)
    if not cases:
        sys.exit(f"测试集为空或格式不正确: {args.set}")

    print(f"\n加载 {len(cases)} 条用例 | 后端={args.backend} | top_k={args.top_k}")
    print(
        "配置: 混合检索=%s 重排=%s 父子块=%s 查询改写=%s Multi-Query=%s\n"
        % (
            settings.ENABLE_HYBRID_SEARCH,
            settings.ENABLE_RERANK,
            settings.ENABLE_PARENT_CHILD,
            settings.ENABLE_QUERY_REWRITE,
            settings.ENABLE_MULTI_QUERY,
        )
    )

    evaluator = Evaluator()
    rows: List[Dict[str, Any]] = []

    print(f"{'用例':<14} {'命中':>5} {'召回':>6} {'MRR':>5} {'忠实':>5} {'相关':>5} {'相似':>5} {'拒答':>5}  耗时")
    print("-" * 82)

    for idx, case in enumerate(cases, start=1):
        row = evaluator.run_case(case, args.top_k)
        rows.append(row)
        print(
            f"{str(row['id']):<14} {fmt(row['hit'])} {fmt(row['recall'])} {fmt(row['mrr'])} "
            f"{fmt(row['faithfulness'])} {fmt(row['relevancy'])} {fmt(row['similarity'])} "
            f"{fmt(row['refusal_correct'])}  "
            f"检索{row['latency_retrieve']:.2f}s 生成{row['latency_generate']:.2f}s"
        )

    print("-" * 82)

    summary = {
        "hit_rate": mean([r["hit"] for r in rows]),
        "recall": mean([r["recall"] for r in rows]),
        "mrr": mean([r["mrr"] for r in rows]),
        "faithfulness": mean([r["faithfulness"] for r in rows]),
        "relevancy": mean([r["relevancy"] for r in rows]),
        "similarity": mean([r["similarity"] for r in rows]),
        "refusal_correct": mean([r["refusal_correct"] for r in rows]),
        "avg_latency_retrieve": mean([r["latency_retrieve"] for r in rows]),
        "avg_latency_generate": mean([r["latency_generate"] for r in rows]),
    }

    print(
        f"{'汇总':<14} {fmt(summary['hit_rate'])} {fmt(summary['recall'])} {fmt(summary['mrr'])} "
        f"{fmt(summary['faithfulness'])} {fmt(summary['relevancy'])} {fmt(summary['similarity'])} "
        f"{fmt(summary['refusal_correct'])}"
    )
    print(
        f"\n平均耗时: 检索 {summary['avg_latency_retrieve']}s / 生成 {summary['avg_latency_generate']}s"
    )

    if args.backend == "ragas":
        print("\n正在计算 RAGAS 官方指标…")
        ragas_scores = run_ragas(rows, evaluator.engine)
        summary["ragas"] = {k: (float(v) if v is not None else None) for k, v in ragas_scores.items()}
        for key, value in ragas_scores.items():
            print(f"  {key:<20} {value}")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "cases": rows}, f, ensure_ascii=False, indent=2)
        print(f"\n报告已写入: {args.output}")

    print()


if __name__ == "__main__":
    main()
