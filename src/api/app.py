import os
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.schemas.contracts import ShiftSafetyManifest, IncidentTriageVerdict
from src.api.demo_service import DemoService

logger = logging.getLogger("SentinelAPI")

app = FastAPI(title="SentinelZone-AI Incident Broker & Interactive Demo Gateway", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Demo Service on GPU
demo_service = DemoService.get_instance()

# Ensure directories exist
media_dir = Path("data/test_videos").resolve()
media_dir.mkdir(parents=True, exist_ok=True)
dashboard_dir = Path("src/dashboard").resolve()

# Mount media video files
app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.get("/")
async def serve_dashboard():
    index_file = dashboard_dir / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Dashboard index.html not found")
    return FileResponse(str(index_file))


@app.get("/health")
async def health_check():
    return {
        "status": "HEALTHY",
        "system": "SentinelZone-AI",
        "version": "4.0.0",
        "device": demo_service.device,
        "active_connections": len(manager.active_connections)
    }


# ==================== Demo API Endpoints ====================

@app.get("/api/v1/demo/scenarios")
def get_demo_scenarios():
    """
    Returns available unseen test scenarios from the real dataset.
    """
    return demo_service.get_scenarios()


@app.get("/api/v1/demo/telemetry/{scenario_id}")
def get_scenario_telemetry(scenario_id: str):
    """
    Returns complete sequential telemetry for the given unseen scenario.
    """
    try:
        return demo_service.get_scenario_telemetry(scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/demo/frame/{scenario_id}/{frame_idx}")
def get_demo_frame(scenario_id: str, frame_idx: int):
    """
    Returns GPU inference telemetry for a specific frame index.
    """
    try:
        return demo_service.process_frame_by_index(scenario_id, frame_idx)
    except IndexError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/demo/frame_image/{scenario_id}/{frame_idx}")
def get_demo_frame_image(scenario_id: str, frame_idx: int):
    """
    Returns the real raw video/dataset frame as a JPEG image.
    Guarantees cross-browser rendering with zero video codec failure.
    """
    try:
        img_bytes = demo_service.get_frame_image_bytes(scenario_id, frame_idx)
        return Response(content=img_bytes, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.websocket("/ws/demo_stream/{scenario_id}")
async def websocket_demo_stream(websocket: WebSocket, scenario_id: str):
    """
    Live streaming telemetry WebSocket for real-time synchronized playback.
    """
    await websocket.accept()
    try:
        telemetry = demo_service.get_scenario_telemetry(scenario_id)
        for frame_data in telemetry:
            await websocket.send_json(frame_data)
            await asyncio.sleep(0.05)  # 20 FPS streaming rate
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))


# ==================== Incident & Adjudication Endpoints ====================

@app.websocket("/ws/incidents")
async def websocket_incident_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.post("/api/v1/incidents/publish")
async def publish_edge_incident(incident_packet: dict):
    await manager.broadcast(incident_packet)
    return {"status": "BROADCAST_SUCCESS", "received_event": incident_packet.get("event_id")}


@app.post("/api/v1/adjudication/submit")
async def submit_human_adjudication(verdict: IncidentTriageVerdict):
    return {
        "status": "ADJUDICATION_STORED",
        "event_id": verdict.event_id,
        "recorded_verdict": verdict.verdict,
        "retraining_priority": verdict.retraining_priority
    }


# ==================== Autonomous Agentic Endpoints ====================

class AgentChatRequest(BaseModel):
    message: str
    site_id: Optional[str] = "SITE-01"
    history: Optional[List[Dict[str, str]]] = None


class CompileManifestRequest(BaseModel):
    permit_text: str
    site_id: Optional[str] = "SITE-01"
    shift_date: Optional[str] = "2026-09-07"
    ifc_file_path: Optional[str] = "models/site_pier_b4.ifc"


class AdjudicateRequest(BaseModel):
    event_id: Optional[str] = None
    video_s3_uri: Optional[str] = "s3://sentinel-incidents/scenario_worker_in_excavator_blind_spot.mp4"
    telemetry_json: Optional[Dict[str, Any]] = None
    shift_context: Optional[str] = None


@app.post("/api/v1/agent/supervisor/chat")
async def supervisor_agent_chat(req: AgentChatRequest):
    """
    Direct multi-turn interaction with SentinelZone-AI's Autonomous Site Safety Supervisor ReAct agent.
    Autonomously invokes real domain tools (telemetry, permit compilation, adjudication, toolbox briefing, etc.).
    """
    try:
        from src.graph.supervisor_agent import build_supervisor_agent
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        
        messages = []
        if req.history:
            for item in req.history:
                role = item.get("role")
                content = item.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content=req.message))

        agent = build_supervisor_agent()
        result = agent.invoke({
            "messages": messages,
            "site_id": req.site_id or "SITE-01",
            "current_step": 0
        })

        out_messages = result.get("messages", [])
        final_text = ""
        tool_traces = []

        for m in out_messages:
            if isinstance(m, ToolMessage):
                tool_traces.append({
                    "tool": m.name,
                    "output": m.content
                })
            elif isinstance(m, AIMessage):
                if m.content:
                    final_text = m.content

        return {
            "response": final_text,
            "tool_executions": tool_traces,
            "site_id": req.site_id,
            "total_steps": result.get("current_step", 1)
        }
    except Exception as e:
        logger.error(f"Error in supervisor_agent_chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agent/pipeline1/compile_manifest")
async def pipeline1_compile_manifest(req: CompileManifestRequest):
    """
    Executes Pipeline 1: Natural language PTW extraction -> BIM spatial envelope resolution
    -> ShiftSafetyManifest Pydantic v2 validation -> Edge collision engine injection.
    """
    try:
        from src.graph.context_pipeline import build_context_pipeline
        pipe = build_context_pipeline()
        res = pipe.invoke({
            "site_id": req.site_id or "SITE-01",
            "shift_date": req.shift_date or "2026-09-07",
            "raw_permit_text": req.permit_text,
            "ifc_file_path": req.ifc_file_path or "models/site_pier_b4.ifc",
            "extracted_tasks": [],
            "resolved_envelopes": [],
            "validated_manifest": None,
            "validation_errors": [],
            "dispatch_status": ""
        })

        manifest = res.get("validated_manifest")
        return {
            "shift_id": manifest.shift_id if manifest else None,
            "site_id": req.site_id,
            "tasks_count": len(res.get("extracted_tasks", [])),
            "envelopes_count": len(res.get("resolved_envelopes", [])),
            "dispatch_status": res.get("dispatch_status"),
            "manifest": manifest.model_dump() if manifest else None,
            "validation_errors": res.get("validation_errors", [])
        }
    except Exception as e:
        logger.error(f"Error in pipeline1_compile_manifest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/agent/pipeline2/adjudicate")
async def pipeline2_adjudicate(req: AdjudicateRequest):
    """
    Executes Pipeline 2: Multimodal VLM Triage -> Active Learning Curation -> Forensic Archival.
    """
    try:
        import uuid
        from src.graph.adjudication_pipeline import build_adjudication_pipeline
        evt_id = req.event_id or f"INC-{uuid.uuid4().hex[:8].upper()}"
        pipe = build_adjudication_pipeline()
        res = pipe.invoke({
            "event_id": evt_id,
            "video_s3_uri": req.video_s3_uri or f"s3://sentinel-incidents/{evt_id}.mp4",
            "telemetry_json": req.telemetry_json or {"min_ttc": 1.4, "p_col": 0.88},
            "shift_context": req.shift_context or "Active excavation work zone",
            "final_verdict": None,
            "active_learning_curated": False,
            "archive_path": None
        })

        v = res.get("final_verdict")
        return {
            "event_id": evt_id,
            "verdict": v.model_dump() if v else None,
            "active_learning_curated": res.get("active_learning_curated", False),
            "archive_path": res.get("archive_path")
        }
    except Exception as e:
        logger.error(f"Error in pipeline2_adjudicate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/toolbox_talk/{site_id}")
async def get_toolbox_talk(site_id: str, focus_hazard: Optional[str] = None):
    """
    Synthesizes an OSHA-compliant daily safety toolbox briefing (29 CFR 1926).
    """
    try:
        from src.graph.supervisor_agent import generate_daily_toolbox_talk
        raw = generate_daily_toolbox_talk.invoke({"site_id": site_id, "focus_hazard": focus_hazard})
        import json
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/agent/active_learning/queue")
async def get_curated_active_learning_queue():
    """
    Returns statistics and curated edge-case dataset samples ready for retraining.
    """
    try:
        from src.graph.supervisor_agent import get_active_learning_queue
        raw = get_active_learning_queue.invoke({})
        import json
        return json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
