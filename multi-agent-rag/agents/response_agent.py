# agents/response_agent.py
from config import client, MODEL_SMART

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided document context.

Rules:
- Base every statement on the context. Do not add facts from outside it.
- Cite the supporting context block after each claim using its number, like [1] or [2][3].
- If different blocks agree, combine them into one clear answer; if they conflict, say so.
- If the context only partly answers the question, answer the part it covers and state plainly what is missing.
- If the context does not answer the question at all, say so instead of guessing.
- Lead with the direct answer, then give the key explanation. Use short paragraphs, and a short list only when comparing items or listing steps.
- Write plain text. Do not use markdown symbols such as **, #, or backticks.
- Use the earlier conversation only to understand follow-up questions, never as a source of facts."""


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    if not context_chunks:
        return query
    # Numbered blocks so the model can cite [n]; best chunk first.
    context = "\n\n".join(
        f"[{i}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(context_chunks, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def _messages(query: str, context_chunks: list[dict], history: list[dict] | None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # Last 2 exchanges let "explain more" / "what about X?" make sense.
        messages.extend(
            {"role": t["role"], "content": t["content"][:1500]} for t in history[-4:]
        )
    messages.append({"role": "user", "content": build_prompt(query, context_chunks)})
    return messages


class ResponseAgent:
    def run(self, query: str, context_chunks: list[dict], history: list[dict] | None = None) -> str:
        if not context_chunks:
            return "I couldn't find relevant information to answer your question."

        response = client.chat.completions.create(
            model=MODEL_SMART,
            messages=_messages(query, context_chunks, history),
            max_tokens=1500,
            temperature=0.2,   # factual, grounded answers; less drift from the context
        )
        return response.choices[0].message.content or ""

    def stream(self, query: str, context_chunks: list[dict], history: list[dict] | None = None):
        """Yields text tokens one by one."""
        if not context_chunks:
            yield "I couldn't find relevant information to answer your question."
            return

        completion = client.chat.completions.create(
            model=MODEL_SMART,
            messages=_messages(query, context_chunks, history),
            max_tokens=1500,
            temperature=0.2,
            stream=True,
        )

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                yield token
