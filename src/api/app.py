import os
import asyncio
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.schemas.contracts import ShiftSafetyManifest, IncidentTriageVerdict
from src.api.demo_service import DemoService

app = FastAPI(title="SentinelZone-AI Incident Broker & Interactive Demo Gateway", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Demo Service on GPU
demo_service = DemoService()

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
