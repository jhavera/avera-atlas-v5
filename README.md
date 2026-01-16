# AVERA-ATLAS

**Advanced Virtualized Enterprise Reconfigurable Architecture – Advanced Tracking and Location Analysis System**

On-orbit debris detection and collision risk assessment system using SWIR imagery, AI-based detection, multi-sensor fusion, Initial Orbit Determination (IOD), NASA-standard probability of collision (Pc) calculations, and automated maneuver decision support.

## Overview

AVERA-ATLAS is a microservices-based system designed for edge deployment on spacecraft (NVIDIA Jetson) or ground-based operations. It processes sensor imagery to detect orbital debris, correlates detections across multiple sensors, performs Initial Orbit Determination, propagates trajectories, computes collision probability, and provides GO/NO-GO maneuver recommendations based on the decision framework described in Kevin Sampson's "Thoughts on MMOD Risk & Propulsion System Sizing."

**Key Capabilities:**
- Real-time debris detection from SWIR imagery using YOLO/ONNX inference
- Multi-sensor detection correlation and track management
- Initial Orbit Determination (IOD) using Gauss and Herrick-Gibbs methods
- Keplerian orbit propagation for trajectory prediction
- NASA-standard Pc calculation with Frisbee max-Pc for missing covariance cases
- Risk-tiered alerting (RED/AMBER/GREEN) based on industry thresholds
- GO/NO-GO decision matrix with time-to-TCA analysis
- Propulsion system recommendations (Option A high-thrust vs Option B high-efficiency)
- 3D visualization with Pc overlay and decision status
- Enhanced operator dashboard with real-time decision support
- Demo mode for autonomous operation and investor presentations

## Architecture

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│  Detector   │─────▶│   Tracker   │─────▶│ Propagator  │
│ (YOLO/ONNX) │      │ (IOD/Fusion)│      │  (Kepler)   │
└─────────────┘      └─────────────┘      └─────────────┘
     :8000                :8000                  │
                                                 │
                                                 ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│     UI      │◀─────│     Viz     │◀─────│  Artifacts  │
│ (Dashboard) │      │ (Matplotlib)│      │   (.npz)    │
└─────────────┘      └─────────────┘      └─────────────┘
     :8080

Data Flow:
  detections ──▶ states_multi.npz ──▶ prop_multi.npz ──▶ planner_output.mp4
```

**Analytical Planning Stack (APS) Pipeline:**
1. **Detection** - SWIR imagery processed by YOLO model to identify debris/satellites
2. **Tracking** - Multi-sensor correlation, UCT management, Initial Orbit Determination
3. **Propagation** - Keplerian orbit propagation with conjunction screening
4. **Assessment** - NASA-standard Pc calculation, risk classification
5. **Decision** - GO/NO-GO recommendation with propulsion options
6. **Visualization** - 3D trajectory rendering and operator dashboard

**Services:**

| Service | Port | Description |
|---------|------|-------------|
| detector | 8000 | SWIR image processing, YOLO inference |
| tracker | 8000 | Multi-sensor fusion, correlation, IOD, track management |
| propagator | - | Keplerian propagation, conjunction screening, Pc calculation |
| viz | - | 3D trajectory rendering with risk coloring |
| ui | 8080 | Web dashboard with decision support and demo mode |

## Quick Start

```bash
# Clone and enter directory
cd avera-atlas-v5

# Build all services
docker-compose build

# Run the system
docker-compose up
```

Access the UI at http://localhost:8080

### Demo Mode

For demonstrations without real sensor data:
1. Open the UI at http://localhost:8080
2. Click **"▶ START DEMO MODE"** in the SWIR Data Ingest panel
3. The system will automatically generate synthetic detections every 15 seconds
4. Watch the Tracker Pipeline portlet show detection → correlation → IOD → track flow
5. Conjunction assessments and trajectory visualizations update automatically

## Project Structure

```
avera-atlas-v5/
├── docker-compose.yaml              # Orchestrates all services
├── README.md                        # This file
├── swir-detector/                   # Debris detection service
│   ├── main.py                      # FastAPI + ONNX inference
│   ├── detector_yolo.py             # YOLO detection module
│   └── Dockerfile
├── tracker-service/                 # Multi-sensor fusion & IOD
│   ├── main.py                      # FastAPI tracker endpoints
│   ├── correlate.py                 # Detection correlation engine
│   ├── iod.py                       # Initial Orbit Determination
│   ├── transform.py                 # Sensor-to-inertial transforms
│   ├── mock_platform.py             # Simulated platform states
│   ├── models.py                    # Data models
│   ├── schemas.py                   # API schemas
│   ├── README.md                    # Tracker-specific documentation
│   └── Dockerfile
├── planner-stack/                   # Core analysis pipeline
│   ├── ingest/
│   │   └── tracks_provider.py       # Detection buffering
│   ├── propagate/
│   │   ├── pc_utils.py              # NASA Pc calculations
│   │   └── propagator_kepler.py     # Propagator with decision engine
│   ├── viz/
│   │   └── render.py                # 3D visualization with decisions
│   ├── data/                        # Shared artifact storage
│   ├── demo_scenarios.py            # Demo conjunction generator
│   ├── Dockerfile.ingest
│   ├── Dockerfile.propagator
│   └── Dockerfile.viz
├── ui/                              # Operator dashboard
│   └── app/
│       ├── main.py                  # FastAPI with decision & demo endpoints
│       └── templates/
│           └── index.html           # Dashboard with tracker status
└── k8s/                             # Kubernetes manifests
```

## Tracker Service

The tracker service bridges detection and propagation, solving the critical problem of converting pixel-level detections into orbital state vectors.

### Pipeline Steps

1. **Detection Ingestion** - Receives detections from SWIR detector
2. **Platform State Retrieval** - Gets observer satellite position/attitude
3. **Sensor-to-Inertial Transform** - Converts pixel coordinates to RA/Dec angles
4. **Detection Correlation** - Associates detections into Uncorrelated Track (UCT) buffers
5. **Initial Orbit Determination** - Computes orbital elements from angular observations
6. **Track Export** - Outputs state vectors in propagator-compatible format

### IOD Methods

- **Gauss Method** - Classical angles-only IOD for 3 observations
- **Herrick-Gibbs** - Velocity estimation from closely-spaced observations
- Automatic method selection based on observation geometry

See `tracker-service/README.md` for detailed tracker documentation.

## Decision Framework

### GO/NO-GO Matrix

Based on Kevin Sampson's recommendation framework, maneuver decisions combine time to TCA (Time of Closest Approach) and probability of collision:

| Time Remaining | Pc = 0.01 | Pc = 0.05 | Pc = 0.10 | Pc = 0.25 |
|----------------|-----------|-----------|-----------|-----------|
| 1 minute       | NO GO     | NO GO     | GO        | GO        |
| 5 minutes      | NO GO     | NO GO     | STANDBY   | GO        |
| 30 minutes     | NO GO     | STANDBY   | STANDBY   | GO        |
| 1 hour         | NO GO     | NO GO     | STANDBY   | STANDBY   |
| 2 hours        | NO GO     | NO GO     | STANDBY   | GO        |

**Decision States:**
- **GO** - Execute collision avoidance maneuver immediately
- **STANDBY** - Prepare thrusters, continue high-fidelity tracking
- **NO GO** - Continue standard monitoring

### Propulsion Options

#### Option A: High Thrust (Late Action)
For maneuvers with **< 30 minutes** to TCA:
- Cold Gas Thrusters (10-1000 mN, Isp: 50-80s)
- Monopropellant (100-5000 mN, Isp: 200-250s)
- High-Power Hall Thrusters (50-500 mN, Isp: 1200-2000s)

#### Option B: High Efficiency (Early Action)
For maneuvers with **> 60 minutes** to TCA:
- Electrospray/Colloid (0.01-1 mN, Isp: 500-2500s)
- FEEP (0.001-0.5 mN, Isp: 4000-10000s)
- Miniature Ion Engines (0.1-5 mN, Isp: 2000-4000s)

## Collision Probability Calculation

The propagator service implements NASA-standard conjunction assessment using the `pc_utils` module.

### Pc Methods

**Standard 2D Pc (`pc_circle`)**: Integrates collision probability over a circular hard-body region on the conjunction plane.

**Frisbee Max-Pc**: When covariance data is unavailable for one object (common for newly detected debris), computes a conservative upper-bound Pc.

### Risk Thresholds

| Level | Pc Threshold | Action |
|-------|--------------|--------|
| 🔴 RED | ≥ 1e-4 | Emergency - maneuver required |
| 🟠 AMBER | ≥ 1e-5 | Watch - continue tracking |
| 🟢 GREEN | ≥ 1e-7 | Monitor |
| NOMINAL | < 1e-7 | No action required |

## UI Dashboard

The operator dashboard includes:

1. **SWIR Data Ingest** - Manual image upload or Demo Mode activation
2. **Tracking Summary** - Total objects tracked, RED/AMBER alert counts
3. **Tracker Pipeline** - Real-time view of detection processing stages, UCT buffers, confirmed tracks
4. **Optical Feed** - Detection visualization with bounding boxes
5. **Active Conjunctions** - Object ID, miss distance, Pc, time to TCA, risk level
6. **Maneuver Decision Panel** - GO/STANDBY/NO GO status with action recommendations
7. **Propulsion Recommendation** - Thruster options with delta-V estimates
8. **GO/NO-GO Reference Matrix** - Quick reference for decision criteria
9. **Trajectory Visualization** - 3D animation and conjunction summary chart

## API Reference

### UI Service (port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/conjunction-status` | GET | Current conjunction assessment with decisions |
| `/api/tracker-status` | GET | Tracker pipeline status, UCTs, tracks |
| `/api/demo/start` | POST | Start demo mode |
| `/api/demo/stop` | POST | Stop demo mode |
| `/api/process` | POST | Process uploaded SWIR image |

### Tracker Service (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Service health and statistics |
| `/detections` | POST | Submit new detection |
| `/tracks` | GET | List all confirmed tracks |
| `/uncorrelated` | GET | List UCT buffers |
| `/export/states` | POST | Export tracks for propagator |
| `/demo/generate` | POST | Generate synthetic detections |

### Detector Service (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Submit SWIR image for detection |
| `/health` | GET | Service health check |

## Data Artifacts

### `states_multi.npz` (Tracker → Propagator)

```python
{
    'object_ids': ['Debris_001', 'CubeSat_001', ...],
    'r_eci_km': [[x, y, z], ...],        # Position vectors (km)
    'v_eci_km_s': [[vx, vy, vz], ...],   # Velocity vectors (km/s)
    'confidences': [0.85, 0.92, ...],    # Track confidence
    't_window': [60.0, 1440],            # [dt_sec, n_steps]
    'metadata': '{"source": "tracker", "t0": "...", "track_count": N}'
}
```

### `prop_multi.npz` (Propagator → Viz/UI)

```python
{
    # Trajectories
    'r_asset': [[x, y, z], ...],
    'r_objects': [[[x, y, z], ...], ...],
    
    # Conjunction Assessment
    'ca_table': [12.5, ...],             # Miss distances (km)
    'pc_values': [3.2e-6, ...],          # Probability of collision
    'risk_levels': ['AMBER', ...],
    
    # Decisions
    'decisions': ['GO', 'STANDBY', ...],
    'propulsion_options': ['A', 'B', ...]
}
```

## Configuration

Key environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data/planner_artifacts` | Shared artifact storage |
| `SWIR_SERVICE_URL` | `http://detector:8000/predict` | Detector endpoint |
| `TRACKER_SERVICE_URL` | `http://tracker:8000` | Tracker endpoint |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## Kubernetes Deployment

```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-detector.yaml
kubectl apply -f k8s/02-planner.yaml
kubectl apply -f k8s/03-viz.yaml
kubectl apply -f k8s/04-ui.yaml
```

## References

- Sampson, K. "Thoughts on MMOD Risk & Propulsion System Sizing"
- Alfano, S. (2005). "A Numerical Implementation of Spherical Object Collision Probability"
- NASA Conjunction Assessment Risk Analysis (CARA) Operations
- Vallado, D. "Fundamentals of Astrodynamics and Applications" (IOD methods)

## Version History

- **v5** - Added tracker service with multi-sensor fusion, IOD, demo mode
- **v4** - Enhanced UI dashboard with decision support
- **v3** - Integrated NASA-standard Pc calculations
- **v2** - Added Keplerian propagation
- **v1** - Initial SWIR detection pipeline

## License

Proprietary - xOrbita Inc.
