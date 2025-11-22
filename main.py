import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Note: Avoid importing optional heavy libs at startup to keep server stable and fast
# NetworkX/spaCy/LangChain can be added later; current graph building uses a lightweight JSON approach

from database import db, create_document

app = FastAPI(title="CogniFloe – Agentic Upgrade Middleware", version="0.1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========================
# Pydantic Models
# ========================
class WorkflowInput(BaseModel):
    text: Optional[str] = Field(None, description="Freeform workflow description")
    domain: Optional[str] = Field(None, description="Industry/domain hint")

class WorkflowOutput(BaseModel):
    workflow_steps: List[Dict[str, Any]]
    decision_points: List[Dict[str, Any]]
    actors: List[str]
    legacy_flowchart_json: Dict[str, Any]
    agent_suggestions: List[Dict[str, Any]]
    agent_blueprints: List[Dict[str, Any]]
    agent_graph: Dict[str, Any]
    automated_percentage: str
    time_saving_estimate: str
    scalability_score: str
    final_agentic_architecture_json: Dict[str, Any]


# ========================
# Helper Functions
# ========================

def simple_nlp_extract(text: str) -> Dict[str, Any]:
    """
    Lightweight rule-based extractor to keep responses fast and offline-ready.
    Identifies steps, decisions, and actors with heuristics.
    """
    sentences = [s.strip(" -•\n\t") for s in text.replace("\r", " ").split(".") if s.strip()]

    steps = []
    decisions = []
    actors = set()

    decision_keywords = {"if", "approve", "reject", "decision", "validate", "check", "review"}
    actor_markers = ["agent", "bot", "system", "analyst", "manager", "customer", "user", "operator"]

    for i, s in enumerate(sentences, start=1):
        for mark in actor_markers:
            if mark in s.lower():
                actors.add(mark.capitalize())
        steps.append({
            "id": f"S{i}",
            "text": s,
        })
        if any(k in s.lower() for k in decision_keywords):
            decisions.append({
                "id": f"D{i}",
                "text": s,
            })

    return {
        "steps": steps,
        "decisions": decisions,
        "actors": sorted(list(actors)) or ["Human", "System"],
    }


def identify_agents(extracted: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggestions = []
    for step in extracted["steps"]:
        txt = step["text"].lower()
        if any(k in txt for k in ["enter", "copy", "transfer", "update", "upload", "extract"]):
            suggestions.append({
                "name": "Automation Agent",
                "role": "Automates repetitive data entry and transfer",
                "capabilities": ["RPA-like actions", "Form filling", "System-to-system sync"],
                "depends_on": [],
                "targets": [step["id"]],
            })
        if any(k in txt for k in ["approve", "review", "validate", "check", "decision"]):
            suggestions.append({
                "name": "Decision Agent",
                "role": "Assists human approvals with policy/risk checks",
                "capabilities": ["Policy rules", "Risk scoring", "Explainable recommendations"],
                "depends_on": [],
                "targets": [step["id"]],
            })
        if any(k in txt for k in ["calculate", "score", "classify", "forecast", "optimize"]):
            suggestions.append({
                "name": "Reasoning Agent",
                "role": "Performs complex calculations and reasoning",
                "capabilities": ["LLM/ReAct", "Tool-use", "Math/analysis"],
                "depends_on": [],
                "targets": [step["id"]],
            })
    if extracted["steps"]:
        suggestions.append({
            "name": "Orchestrator Agent",
            "role": "Coordinates multi-step workflows and error handling",
            "capabilities": ["Routing", "Retries", "Parallelization"],
            "depends_on": [s["name"] for s in suggestions],
            "targets": [s["id"] for s in extracted["steps"]],
        })
    sigs = set()
    dedup = []
    for s in suggestions:
        key = (s["name"], tuple(s["targets"]))
        if key not in sigs:
            sigs.add(key)
            dedup.append(s)
    return dedup


def build_graph(steps: List[Dict[str, Any]], decisions: List[Dict[str, Any]], agents: List[Dict[str, Any]]):
    # Lightweight JSON graph builder (no external deps)
    nodes = []
    edges = []
    for s in steps:
        nodes.append({"id": s["id"], "label": s["text"], "type": "step"})
    for d in decisions:
        nodes.append({"id": d["id"], "label": d["text"], "type": "decision"})
    for i in range(len(steps) - 1):
        edges.append({"source": steps[i]["id"], "target": steps[i+1]["id"], "type": "flow"})
    for idx, a in enumerate(agents):
        aid = f"A{idx+1}"
        nodes.append({"id": aid, "label": a["name"], "type": "agent"})
        for t in a["targets"]:
            edges.append({"source": aid, "target": t, "type": "assists"})
    return {"nodes": nodes, "edges": edges}


def compute_metrics(steps_count: int, agents_count: int) -> Dict[str, str]:
    auto_pct = min(90, 20 + agents_count * 10)
    time_saved = min(80, 10 + agents_count * 8)
    scale_score = min(95, 50 + agents_count * 7)
    return {
        "automated_percentage": f"{auto_pct}%",
        "time_saving_estimate": f"{time_saved}%",
        "scalability_score": f"{scale_score}/100",
    }


def build_flowchart_json(steps: List[Dict[str, Any]], decisions: List[Dict[str, Any]]):
    return {
        "steps": steps,
        "decisions": decisions,
        "edges": [
            {"source": steps[i]["id"], "target": steps[i+1]["id"], "label": "next"}
            for i in range(len(steps) - 1)
        ],
    }


# ========================
# Routes
# ========================
@app.get("/")
def read_root():
    return {"message": "CogniFloe Backend Ready"}


@app.post("/api/parse", response_model=WorkflowOutput)
def parse_workflow(payload: WorkflowInput):
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="No workflow text provided")

    extracted = simple_nlp_extract(payload.text)
    agents = identify_agents(extracted)
    graph_json = build_graph(extracted["steps"], extracted["decisions"], agents)
    flowchart_legacy = build_flowchart_json(extracted["steps"], extracted["decisions"])
    metrics = compute_metrics(len(extracted["steps"]), len(agents))

    output: WorkflowOutput = WorkflowOutput(
        workflow_steps=extracted["steps"],
        decision_points=extracted["decisions"],
        actors=extracted["actors"],
        legacy_flowchart_json=flowchart_legacy,
        agent_suggestions=agents,
        agent_blueprints=[{
            "name": a["name"],
            "inputs": "Signals from upstream steps",
            "outputs": "Decisions/Actions",
            "triggers": "When target step is reached",
        } for a in agents],
        agent_graph=graph_json,
        automated_percentage=metrics["automated_percentage"],
        time_saving_estimate=metrics["time_saving_estimate"],
        scalability_score=metrics["scalability_score"],
        final_agentic_architecture_json=graph_json,
    )

    try:
        create_document("analysis", {
            "domain": payload.domain or "generic",
            "text": payload.text[:1000],
            "metrics": metrics,
            "agents": [a["name"] for a in agents],
        })
    except Exception:
        pass

    return output


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["connection_status"] = "Connected"
            try:
                response["collections"] = db.list_collection_names()[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
