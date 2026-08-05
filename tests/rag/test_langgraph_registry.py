from pathlib import Path
import json


def test_graphs_registered():
    p = Path("langgraph.json")
    assert p.exists(), "langgraph.json not found"
    data = json.loads(p.read_text(encoding="utf-8"))
    graphs = data.get("graphs") or {}
    assert "rag_ingest" in graphs
    assert "rag_agent" in graphs
