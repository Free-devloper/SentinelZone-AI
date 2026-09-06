import json
import os
import logging
from typing import TypedDict, Dict, Any, Optional
from src.schemas.contracts import IncidentTriageVerdict

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False
    ChatOpenAI = None
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


VLM_ADJUDICATION_PROMPT = """
You are a Lead Construction Safety Auditor. Examine this video telemetry and site context.

Telemetry Data:
{telemetry}

Shift Operating Context:
{context}

Analyze:
1. Was a designated spotter visually directing the machinery?
2. Did the pedestrian worker exhibit awareness (eye contact, hand wave, stopping outside swing radius)?
3. Adjudicate the encounter:
   - TRUE_POSITIVE: Hazardous near-miss or dangerous spatial violation.
   - FALSE_POSITIVE: Perception error, track switch, or spurious alarm.
   - CONTROLLED_WORK: Authorized close operation following standard procedures.

Return ONLY a valid JSON object matching this schema:
{{
  "event_id": "{event_id}",
  "verdict": "TRUE_POSITIVE",
  "confidence": 0.96,
  "spotter_verified": false,
  "worker_awareness_observed": false,
  "root_cause_summary": "Wheel loader reversed into pedestrian crossing path without spotter guidance.",
  "retraining_priority": "HIGH",
  "recommended_mitigation": "Install physical jersey barrier between haul lane and foot transit path."
}}
"""


def _get_vlm():
    if HAS_LANGGRAPH and ChatOpenAI is not None and os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(model="gpt-4o", temperature=0.0)
    return None


def vlm_triage_node(state: IncidentTriageState) -> Dict[str, Any]:
    vlm = _get_vlm()
    if vlm is not None:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", VLM_ADJUDICATION_PROMPT)
            ])
            chain = prompt | vlm
            response = chain.invoke({
                "telemetry": json.dumps(state.get("telemetry_json", {})),
                "context": state.get("shift_context", ""),
                "event_id": state.get("event_id", "UNKNOWN_EVENT")
            })
            data = json.loads(response.content)
            verdict = IncidentTriageVerdict(**data)
            return {"final_verdict": verdict}
        except Exception as e:
            logger.error(f"VLM incident parsing error via LLM: {e}")

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

    builder.set_entry_point("vlm_triage")
    builder.add_edge("vlm_triage", "filter_al")
    builder.add_edge("filter_al", END)
    return builder.compile()
