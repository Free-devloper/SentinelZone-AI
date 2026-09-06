# SentinelZone-AI: Predictive Dynamic Spatial Safety & Near-Miss Anticipation

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-RTX%204070%20Laptop%20%7C%20Jetson%20Orin-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?logo=three.js&logoColor=white)](https://threejs.org)
[![Tests](https://img.shields.io/badge/Tests-33%2F33%20Passing-brightgreen)](tests/)

**SentinelZone-AI** is an edge-grade, predictive dynamic spatial safety and near-miss anticipation platform designed for active industrial construction, mining, and civil infrastructure workzones.

Unlike conventional reactive computer vision systems that simply draw 2D bounding boxes, SentinelZone-AI projects visual detections onto a calibrated geodetic metric ground plane, maintains multi-agent kinematic state tracks via Extended Kalman Filters (EKF), forecasts future trajectories using Spatio-Temporal Graph Attention Networks (GATv2 ST-GNN), dynamically computes Time-to-Collision (TTC) against active machinery blind spots, and streams real-time telemetry to an interactive 3D WebGL Digital Twin.

---

## Architecture Overview

```
                                  PIPELINE ARCHITECTURE
                                  
  +----------------+       +------------------------+       +----------------------+
  | Camera Stream  | --->  | CUDA Perception Engine | --->  | Planar Homography H⁻¹|
  | (RTSP / Video) |       | YOLOv8 + PPE Analysis  |       | (2D Box -> Metric 3D)|
  +----------------+       +------------------------+       +----------+-----------+
                                                                       |
  +--------------------------------------------------------------------+
  |
  v
+------------------+       +------------------------+       +----------------------+
|  Multi-Agent EKF | --->  |   GATv2 ST-GNN Engine  | --->  | Dynamic Exclusion    |
|  State Tracking  |       | Trajectory Forecasting |       | Envelopes & TTC Calc |
+------------------+       +------------------------+       +----------+-----------+
                                                                       |
  +--------------------------------------------------------------------+
  |
  v
+----------------------------------------------------------------------------------+
|                             FASTAPI EDGE PLATFORM                                |
|   * Live Telemetry Stream (/api/v1/demo/telemetry)                               |
|   * Canvas JPEG Stream (Zero Video Codec Failure)                                |
|   * Three.js 3D WebGL Metric Ground Plane Digital Twin                           |
|   * Human-in-the-Loop Incident Adjudication & Active Learning Queue             |
+----------------------------------------------------------------------------------+
```

---

## Core Capabilities

1. **Edge Perception & Multi-Class PPE Compliance:**
   - Detects workers, heavy machinery (excavators, dump trucks, loaders), and static spatial hazards with CUDA hardware acceleration (cuda:0).
   - Evaluates head and torso regions for safety gear compliance (hardhats, high-visibility vests).

2. **Metric Ground Projection via Planar Homography:**
   - Transforms 2D pixel coordinates to real-world ground-plane metric coordinates (in meters) with sub-0.15m RMSE calibration drift monitoring.

3. **Multi-Agent Kinematic Tracking (EKF):**
   - Decoupled Extended Kalman Filtering tracking position, metric velocity, acceleration, and heading angle.

4. **Spatio-Temporal Graph Neural Forecasting (GATv2 ST-GNN):**
   - Encodes historical kinematic sequences with Gated Recurrent Units (GRU) and applies multi-head dynamic edge attention to predict 3.0s future trajectories and anticipate near-miss events before they occur.

5. **Dynamic Exclusion Envelopes & TTC Debouncing:**
   - Computes Time-to-Collision (TTC) using dynamic exclusion polygons that expand and rotate with heavy machinery swing radius and blind spots.
   - Debounces hazard alerts over sliding temporal windows to eliminate false positives.

6. **Interactive 3D WebGL Digital Twin:**
   - Synchronized dual-pane operator dashboard featuring a canvas-based video overlay alongside a Three.js 3D geodetic datum twin.

7. **Dynamic Zero-Configuration Video Ingestion:**
   - Drop any video (.mp4, .avi, .mov, .mkv, .webm) into data/real_videos/ and it is automatically discovered, registered, and processed with zero code changes required.

---

## Repository Structure

`
SentinelZone-AI/
|-- config/
|   +-- default_config.json        # Calibrated homography, thresholds, and envelope specs
|-- data/
|   |-- construction_safety/       # Unit test fixtures and dataset manifests
|   |-- real_videos/               # Real-world site videos (dynamic drop folder)
|   |-- test_videos/               # Calibrated scenario sequences
|   +-- roboflow_downloaded/       # Roboflow validation and test splits
|-- src/
|   |-- api/
|   |   |-- app.py                 # FastAPI application with threadpool routes
|   |   +-- demo_service.py        # Dynamic video scanner, GPU processor, and telemetry cache
|   |-- dashboard/
|   |   +-- index.html             # Three.js 3D Digital Twin & canvas dashboard
|   |-- perception/
|   |   |-- detector.py            # YOLOv8 + PPE multi-class detector
|   |   +-- dataset_downloader.py  # Automated Roboflow dataset downloader
|   |-- spatial/
|   |   |-- homography.py          # Planar homography matrix & calibration engine
|   |   +-- bim_resolver.py        # BIM/IFC geodetic coordinate resolver
|   +-- tracking/
|       |-- ekf.py                 # Kinematic Extended Kalman Filter
|       |-- gnn_forecaster.py      # GATv2 Spatio-Temporal Graph Neural Network
|       +-- conflict_engine.py     # Dynamic exclusion envelopes & debouncer
|-- tests/
|   |-- test_demo_api.py           # API integration tests
|   |-- test_detector.py           # YOLO and PPE tests
|   |-- test_ekf.py                # Kinematic tracking tests
|   |-- test_gnn_forecaster.py     # GATv2 trajectory forecast tests
|   |-- test_homography.py         # Homography projection and drift tests
|   |-- test_incident_agent.py     # Incident reporting tests
|   +-- test_safety_engine.py      # Safety envelopes and debouncing tests
|-- agent.md                       # Comprehensive mathematical derivations and specifications
|-- Dockerfile.edge                # Jetson Orin / edge deployment container
+-- pyproject.toml                 # Project metadata and dependencies
`

---

## Quick Start

### 1. Installation

`ash
# Clone repository
git clone https://github.com/Free-devloper/SentinelZone-AI.git
cd SentinelZone-AI

# Install dependencies
pip install torch torchvision ultralytics fastapi uvicorn opencv-python numpy scipy
`

### 2. Launch the Platform

`ash
# Start FastAPI server with CUDA acceleration
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
`

Open your browser at:
**http://localhost:8000**

### 3. Run Unit Tests

`ash
python -m unittest discover -s tests -p test_*.py
`
*All 33 unit tests execute in ~10 seconds with 100% green status.*

---

## Dynamic Video Ingestion

1. Drop any site video file into:
   `
   data/real_videos/
   `
2. Open the dashboard at http://localhost:8000 and click the **Rescan** button in the header (or restart the server).
3. The video is immediately cataloged and ready for live evaluation.

---

## Mathematical Specification & Theory

For complete mathematical proofs, kinematic formulations, homography matrices, and GATv2 attention derivations, refer to:
[**gent.md**](agent.md)

---

## License

Apache 2.0
