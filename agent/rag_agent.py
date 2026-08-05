"""Conversational RAG agent: a ReAct loop over a single KB-search tool, no CRM/sales
tools. Fixed chunking/search_type/reranker (didactic); the model can only adjust
`match_threshold` on the fly for recall/precision.

Ported from ../ai-agent-sales/app/rag/proposal_agent_v2.py, stripped of its
proposal/CRM framing (it already had no CRM tools — "isolated RAG-only test version"
per its own docstring) and adapted to route the LLM through OpenRouter like the rest
of this repo (rag/ingest.py's get_embedder(), rag/search.py's hyde()) rather than
OpenAI directly.
"""

import os
from typing import Optional

from agent.react import create_react_executor
from rag.search import kb_search_client

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_model():
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY (required for the agent's LLM via OpenRouter)")
    return ChatOpenAI(
        model=os.getenv("RAG_AGENT_LLM_MODEL", "openai/gpt-4.1-mini"),
        temperature=0,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=api_key,
    )


def defaults_kb() -> dict:
    return {
        "search_type": "hybrid_rrf",
        "chunking": "semantic",
        "use_hyde": False,
        "reranker": "none",
        "match_threshold": 0.5,
    }


def system_prompt() -> str:
    d = defaults_kb()
    thr = d.get("match_threshold")
    return (
        "You are a KB search assistant focused on RAG (retrieval-augmented generation).\n"
        "- Use the kb_search tool to retrieve context.\n"
        "- By default (didactic setup), use these parameters on each call (only adjust match_threshold if needed):\n"
        f"  - search_type={d['search_type']}\n"
        f"  - chunking is fixed at {d['chunking']} (already applied by the tool).\n"
        f"  - initial match_threshold {thr}; adjust it dynamically (lower for more coverage, higher for less noise).\n"
        f"  - reranker={d['reranker']}\n"
        "- Call the tool as many times as needed until you have enough evidence.\n"
        "- Base the final answer only on the retrieved evidence; do not make things up.\n"
        "\n"
        "Guidelines to maximize objectivity:\n"
        "- When the question asks for an objective value (e.g. SLA, priority support channel, "
        "maintenance window, payment terms, warranty), answer DIRECTLY with ONE value.\n"
        "- If the answer varies by empresa (X/Y) and the question doesn't specify one, assume "
        "Empresa X as the default and state that briefly (Assumption: Empresa X), then add a short "
        "note with the alternative: 'If Empresa Y: <value>'.\n"
        "- Avoid answering with a question. If you need clarification, answer with the best "
        "evidence first and only then offer to clarify further.\n"
        "- Reproduce terms exactly as they appear in the KB (e.g. '30/60', '12 meses', '99.9%', "
        "'portal do cliente').\n"
        "- Cite KB sources (doc_path#chunk_ix) for each key claim, in the format: '... [file.md#12]'.\n"
        "- If the evidence is insufficient, temporarily lower match_threshold and search again; if "
        "still insufficient, state plainly that there isn't enough evidence in the KB."
    )


def build_kb_tool():
    """Thin tool with fixed chunking; only match_threshold is variable (didactic)."""
    from langchain_core.tools import tool

    defaults = defaults_kb()
    fixed_chunking = defaults["chunking"]
    fixed_search = defaults["search_type"]
    fixed_reranker = defaults["reranker"]

    @tool
    def kb_search(query: str, k: int = 5, match_threshold: Optional[float] = None):
        """Searches the KB with fixed chunking and an adjustable threshold.

        Args:
            query: the user's question
            k: how many results to return (default 5)
            match_threshold: optional similarity cutoff
        """
        args = {
            "query": query,
            "k": k,
            "search_type": fixed_search,
            "chunking": fixed_chunking,
            "reranker": fixed_reranker,
            "use_hyde": False,
        }
        if match_threshold is not None:
            try:
                args["match_threshold"] = float(match_threshold)
            except Exception:
                pass
        return kb_search_client.invoke(args)

    return kb_search


def build_graph():
    tools = [build_kb_tool()]
    return create_react_executor(
        build_model(),
        tools=tools,
        prompt=system_prompt(),
    )


graph = build_graph()
