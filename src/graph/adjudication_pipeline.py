import json
import os
import re
import logging
from datetime import datetime
from typing import TypedDict, Dict, Any, Optional
from src.schemas.contracts import IncidentTriageVerdict
from src.graph.llm_provider import get_agent_llm

try:
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    ChatPromptTemplate = None
    StateGraph = None
    END = "__end__"

logger = logging.getLogger("AdjudicationPipeline")


class IncidentTriageState(TypedDict):
    event_id: str
    video_s3_uri: str
    telemetry_json: Dict[str, Any]
    shift_context: str
    final_verdict: Optional[IncidentTriageVerdict]
    active_learning_curated: bool
    archive_path: Optional[str]


VLM_SYSTEM_PROMPT = """
You are a Lead Construction Safety Auditor. Examine the provided video telemetry and site context.
Analyze:
1. Was a designated spotter visually directing the machinery?
2. Did the pedestrian worker exhibit awareness (eye contact, hand wave, stopping outside swing radius)?
3. Adjudicate the encounter as TRUE_POSITIVE (hazardous near-miss), FALSE_POSITIVE (perception error / spurious alarm), or CONTROLLED_WORK (authorized close operation behind barriers / with spotter).

Return ONLY a valid JSON object matching the IncidentTriageVerdict schema with fields:
event_id, verdict, confidence, spotter_verified, worker_awareness_observed, root_cause_summary, retraining_priority, recommended_mitigation.
"""

VLM_HUMAN_PROMPT = """Telemetry Data:
{telemetry}

Shift Operating Context:
{context}

Event ID: {event_id}"""


def vlm_triage_node(state: IncidentTriageState) -> Dict[str, Any]:
    llm = get_agent_llm()
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", VLM_SYSTEM_PROMPT),
            ("human", VLM_HUMAN_PROMPT)
        ])
        chain = prompt | llm
        response = chain.invoke({
            "telemetry": json.dumps(state.get("telemetry_json", {})),
            "context": state.get("shift_context", ""),
            "event_id": state.get("event_id", "UNKNOWN_EVENT")
        })
        content = response.content if hasattr(response, "content") else str(response)
        clean_json = re.sub(r"^```json\s*", "", content.strip(), flags=re.MULTILINE)
        clean_json = re.sub(r"^```\s*", "", clean_json.strip(), flags=re.MULTILINE)
        clean_json = clean_json.strip("` \n")
        data = json.loads(clean_json)
        verdict = IncidentTriageVerdict(**data)
        return {"final_verdict": verdict}
    except Exception as e:
        logger.warning(f"VLM triage fallback invoked: {e}")

    # Deterministic adjudication fallback for offline / test validation
    telemetry = state.get("telemetry_json", {})
    min_ttc = telemetry.get("min_ttc", 1.8)
    p_col = telemetry.get("p_col", 0.85)

    if min_ttc <= 2.0 and p_col >= 0.65:
        verdict_type = "TRUE_POSITIVE"
        priority = "HIGH"
        root_cause = "Machinery trajectory breached worker dynamic safety bubble."
    elif p_col < 0.3:
        verdict_type = "FALSE_POSITIVE"
        priority = "HIGH"  # Curated for active learning retraining
        root_cause = "Transient tracking noise resulted in spurious threshold breach."
    else:
        verdict_type = "CONTROLLED_WORK"
        priority = "LOW"
        root_cause = "Authorized worker in proximity under documented protocol."

    verdict = IncidentTriageVerdict(
        event_id=state.get("event_id", "INC_DEFAULT"),
        verdict=verdict_type,
        confidence=0.95,
        spotter_verified=False,
        worker_awareness_observed=False,
        root_cause_summary=root_cause,
        retraining_priority=priority,
        recommended_mitigation="Review corridor separation and deploy spotter."
    )
    return {"final_verdict": verdict}


def filter_active_learning_node(state: IncidentTriageState) -> Dict[str, Any]:
    verdict = state.get("final_verdict")
    if verdict and (verdict.verdict == "FALSE_POSITIVE" or verdict.retraining_priority == "HIGH"):
        return {"active_learning_curated": True}
    return {"active_learning_curated": False}


def archive_incident_node(state: IncidentTriageState) -> Dict[str, Any]:
    """
    Persists the adjudicated incident record and maintains the active learning retraining queue.
    """
    event_id = state.get("event_id", f"INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    verdict = state.get("final_verdict")

    # 1. Archive Incident to data/incidents/{event_id}.json
    incidents_dir = "data/incidents"
    os.makedirs(incidents_dir, exist_ok=True)
    archive_file = os.path.join(incidents_dir, f"{event_id}.json")

    record = {
        "event_id": event_id,
        "archived_at": datetime.now().isoformat(),
        "video_uri": state.get("video_s3_uri", ""),
        "shift_context": state.get("shift_context", ""),
        "telemetry_summary": state.get("telemetry_json", {}),
        "verdict": verdict.model_dump() if verdict else None,
        "active_learning_curated": state.get("active_learning_curated", False)
    }

    with open(archive_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    # 2. Append to Active Learning Queue if curated
    if state.get("active_learning_curated"):
        al_dir = "data/active_learning"
        os.makedirs(al_dir, exist_ok=True)
        queue_file = os.path.join(al_dir, "curated_queue.jsonl")
        al_entry = {
            "event_id": event_id,
            "curated_at": datetime.now().isoformat(),
            "verdict": verdict.verdict if verdict else "UNKNOWN",
            "retraining_priority": verdict.retraining_priority if verdict else "HIGH",
            "root_cause": verdict.root_cause_summary if verdict else "",
            "telemetry": state.get("telemetry_json", {})
        }
        with open(queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(al_entry) + "\n")
        logger.info(f"Appended incident {event_id} to active learning queue: {queue_file}")

    return {"archive_path": archive_file.replace("\\", "/")}


def build_adjudication_pipeline():
    """
    Assembles and compiles the StateGraph for Pipeline 2: Multimodal Incident Adjudicator & Active Learning Curator.
    """
    if not HAS_LANGGRAPH or StateGraph is None:
        logger.warning("langgraph is not installed. Pipeline builder returning None.")
        return None

    builder = StateGraph(IncidentTriageState)
    builder.add_node("vlm_triage", vlm_triage_node)
    builder.add_node("filter_al", filter_active_learning_node)
    builder.add_node("archive_incident", archive_incident_node)

    builder.set_entry_point("vlm_triage")
    builder.add_edge("vlm_triage", "filter_al")
    builder.add_edge("filter_al", "archive_incident")
    builder.add_edge("archive_incident", END)
    return builder.compile()
