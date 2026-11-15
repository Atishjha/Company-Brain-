# agents/eval_agent.py
"""
LLM-as-judge. Runs in a background thread after the answer is sent, so it
never adds latency. Scores are written back to the query log via a callback.
main.py already imports and calls this class.
"""
import threading

from config import client, MODEL_FAST, parse_json_object

SYSTEM_PROMPT = """You are a strict evaluator of retrieval-augmented answers.

Given a question, the context passages the answer was built from, and the answer, score 0-10:
- faithfulness: every claim in the answer is supported by the context (10 = fully supported, 0 = invented)
- relevance: the answer addresses what was actually asked
- completeness: all parts of the question are covered, as far as the context allows

Return ONLY a JSON object:
{"faithfulness": 8, "relevance": 9, "completeness": 7}"""


class EvalAgent:
    def evaluate(self, query: str, rewritten_query: str, context_chunks: list[dict], answer: str) -> dict | None:
        if not answer.strip() or not context_chunks:
            return None
        context = "\n\n".join(
            f"[{i}] {c['text'][:1200]}" for i, c in enumerate(context_chunks, start=1)
        )
        prompt = f"Question: {query}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
        try:
            resp = client.chat.completions.create(
                model=MODEL_FAST,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=80,
                temperature=0.0,
            )
            raw = parse_json_object(resp.choices[0].message.content or "")
            scores = {}
            for key in ("faithfulness", "relevance", "completeness"):
                scores[key] = max(0.0, min(10.0, float(raw[key])))
            return scores
        except Exception as e:
            print(f"[Eval] Error: {e}")
            return None

    def evaluate_async(self, query, rewritten_query, context_chunks, answer, callback):
        def _work():
            scores = self.evaluate(query, rewritten_query, context_chunks, answer)
            if scores:
                callback(scores)
        threading.Thread(target=_work, daemon=True).start()
