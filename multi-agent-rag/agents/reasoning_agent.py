# agents/reasoning_agent.py
from config import client, MODEL_SMART, parse_json_array

MIN_SCORE     = 6    # keep chunks scoring at least this (0-10)
FALLBACK_KEEP = 2    # if nothing passes, keep this many top chunks instead of none


class ReasoningAgent:
    def run(self, query: str, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        formatted = "\n\n".join(
            f"[Chunk {i+1}] source: {c.get('source', '?')} | "
            f"rerank_score: {c.get('rerank_score', 'n/a')}\n{c['text'][:1500]}"
            for i, c in enumerate(chunks)
        )

        prompt = f"""You are a relevance reasoning agent.

Given the user query and retrieved document chunks, return ONLY a JSON array:
[
  {{"chunk_index": 1, "relevance_score": 8, "reason": "..."}},
  ...
]

Score each chunk 0-10 for how directly it helps answer the query.
Give a high score only if the chunk contains information needed for the answer.
No other text, just the JSON array.

User query: {query}

Retrieved chunks:
{formatted}"""

        try:
            response = client.chat.completions.create(
                model=MODEL_SMART,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
            )
            scores = parse_json_array(response.choices[0].message.content or "")

            valid = [
                s for s in scores
                if isinstance(s.get("chunk_index"), int)
                and 1 <= s["chunk_index"] <= len(chunks)
            ]
            valid.sort(key=lambda s: s.get("relevance_score", 0), reverse=True)
            kept = [s for s in valid if s.get("relevance_score", 0) >= MIN_SCORE]

            # Never hand the response agent an empty context when the
            # reranker already judged some chunks relevant.
            if not kept:
                kept = valid[:FALLBACK_KEEP]

            return [chunks[s["chunk_index"] - 1] for s in kept]

        except Exception as e:
            # Previously a malformed JSON reply crashed the whole request.
            print(f"[Reasoning] Error: {e} — passing reranked chunks through")
            return chunks[:4]
