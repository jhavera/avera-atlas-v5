# AVERA-ATLAS

**Advanced Virtualized Enterprise Reconfigurable Architecture – Advanced Tracking and Location Analysis System**

On-orbit debris detection and collision risk assessment system using SWIR imagery, AI-based detection, NASA-standard probability of collision (Pc) calculations, and automated maneuver decision support.

## Overview

AVERA-ATLAS is a microservices-based system designed for edge deployment on spacecraft (NVIDIA Jetson) or ground-based operations. It processes sensor imagery to detect orbital debris, propagates trajectories, computes collision probability, and provides GO/NO-GO maneuver recommendations based on the decision framework described in Kevin Sampson's "Thoughts on MMOD Risk & Propulsion System Sizing."

**Key Capabilities:**
- Real-time debris detection from SWIR imagery using YOLO/ONNX inference
- Keplerian orbit propagation for trajectory prediction
- NASA-standard Pc calculation with Frisbee max-Pc for missing covariance cases
- Risk-tiered alerting (RED/AMBER/GREEN) based on industry thresholds
- GO/NO-GO decision matrix with time-to-TCA analysis
- Propulsion system recommendations (Option A high-thrust vs Option B high-efficiency)
- 3D visualization with Pc overlay and decision status
- Enhanced operator dashboard with real-time decision support

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Detector  │────▶│   Ingest    │────▶│    Propagator +     │────▶│     Viz     │
│  (YOLO/ONNX)│     │  (FastAPI)  │     │  Decision Engine    │     │ (Matplotlib)│
└─────────────┘     └─────────────┘     └─────────────────────┘     └─────────────┘
     :8000              :8000                                              │
                            │                                              │
                            ▼                                              ▼
                    states_multi.npz ──────────────────────▶     prop_multi.npz
                                                                      │
                                                                      ▼
                                                              planner_output.mp4
                                                                      │
                                                                      ▼
                                                              ┌─────────────┐
                                                              │ Enhanced UI │
                                                              │  Dashboard  │
                                                              └─────────────┘
                                                                   :8080
```

**Services:**

| Service | Port | Description |
|---------|------|-------------|
| detector | 8000 | SWIR image processing, YOLO inference |
| ingest | 8000 | Buffers detections, writes state vectors |
| propagator | - | Keplerian propagation, conjunction screening, Pc calculation, decision engine |
| viz | - | 3D trajectory rendering with risk coloring |
| ui | 8080 | Web dashboard with decision support display |

## Quick Start

```bash
# Clone and enter directory
cd avera-atlas

# Build all services
docker-compose build

# Run the system
docker-compose up
```

Access the UI at http://localhost:8080

## Project Structure

```
avera-atlas/
├── docker-compose.yaml              # Orchestrates all services
├── README.md                        # This file
├── swir-detector/                   # Debris detection service
│   ├── main.py                      # FastAPI + ONNX inference
│   ├── detector_yolo.py             # YOLO detection module
│   ├── debris_model.onnx            # YOLO model weights
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
│       ├── main.py                  # FastAPI with decision endpoints
│       └── templates/
│           └── index.html           # Dashboard with decision display
└── k8s/                             # Kubernetes manifests
```

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

**Characteristics:** Fast response, higher fuel consumption, suitable for emergency scenarios.

#### Option B: High Efficiency (Early Action)
For maneuvers with **> 60 minutes** to TCA:
- Electrospray/Colloid (0.01-1 mN, Isp: 500-2500s)
- FEEP (0.001-0.5 mN, Isp: 4000-10000s)
- Miniature Ion Engines (0.1-5 mN, Isp: 2000-4000s)

**Characteristics:** Lower thrust, excellent fuel efficiency, suitable for planned maneuvers.

## Collision Probability Calculation

The propagator service implements NASA-standard conjunction assessment using the `pc_utils` module.

### Pc Methods

**Standard 2D Pc (`pc_circle`)**: Integrates collision probability over a circular hard-body region on the conjunction plane. Supports multiple estimation modes:
- Mode 64 (default): 64th-order Gauss-Chebyshev quadrature
- Mode 0: Equal-area square approximation (fast)
- Mode -1: Circumscribing square upper bound
- Mode 1: Full numerical integration (scipy.integrate.quad)

**Frisbee Max-Pc**: When covariance data is unavailable for one object (common for newly detected debris), computes a conservative upper-bound Pc. This ensures safety-critical decisions aren't undermined by missing data.

### Risk Thresholds

| Level | Pc Threshold | Action |
|-------|--------------|--------|
| 🔴 RED | ≥ 1e-4 | Emergency - maneuver required |
| 🟠 AMBER | ≥ 1e-5 | Watch - continue tracking |
| 🟢 GREEN | ≥ 1e-7 | Monitor |
| NOMINAL | < 1e-7 | No action required |

These thresholds align with NASA Conjunction Assessment Risk Analysis (CARA) standards.

### Covariance Handling

The system generates assumed covariances when tracking data is unavailable:

**Detection-based covariance**: Scales with range and inverse detection confidence. Objects detected with low confidence get larger (more conservative) uncertainty ellipsoids.

**SGP4 propagation covariance**: Models TLE accuracy degradation over time (approximately 1 km base + 1.5 km/day growth in the along-track direction).

## Data Artifacts

### Input: `states_multi.npz`

Written by the ingest service with detected object states:

```python
{
    'object_ids': ['DEB_001', 'DEB_002', ...],  # Object identifiers
    'r_eci_km': [[x, y, z], ...],               # Position vectors (km)
    'v_eci_km_s': [[vx, vy, vz], ...],          # Velocity vectors (km/s)
    't_window': [dt_sec, n_steps],              # Propagation window
    'metadata': '{"t0": "2024-01-15T12:00:00Z"}',
    'confidences': [0.95, 0.82, ...]            # Detection confidences (optional)
}
```

### Output: `prop_multi.npz`

Written by the propagator with trajectories, risk assessment, and decisions:

```python
{
    # Trajectories
    't_array': [...],                     # Julian dates
    'r_asset': [[x, y, z], ...],          # Asset trajectory (km)
    'r_objects': [[[x, y, z], ...], ...], # Debris trajectories (km)
    'obj_ids': ['DEB_001', ...],
    
    # Conjunction Assessment
    'ca_table': [12.5, 45.2, ...],        # Miss distances at TCA (km)
    'pc_values': [3.2e-6, 1.1e-8, ...],   # Probability of collision
    'risk_levels': ['AMBER', 'NOMINAL', ...],
    'tca_indices': [142, 89, ...],        # Time index of closest approach
    'relative_velocities': [14.2, 12.8, ...],  # Relative velocity at TCA (km/s)
    
    # Decision Outputs
    'decisions': ['GO', 'STANDBY', 'NO_GO', ...],
    'decision_urgencies': ['CRITICAL', 'HIGH', 'LOW', ...],
    'propulsion_options': ['A', 'B', 'N/A', ...],
    'delta_v_estimates': [0.125, 0.0, 0.0, ...],  # m/s
    
    # Summary
    'screening_params': '{"hbr_m": 15.0, ...}',
    'n_red_alerts': 1,
    'n_amber_alerts': 2,
    'n_go_decisions': 1,
    'n_standby_decisions': 2
}
```

## Configuration

Key parameters in `propagator_kepler.py`:

```python
HBR_M = 15.0                    # Combined hard body radius (meters)
SCREENING_THRESHOLD_KM = 100.0  # Only compute Pc if miss < threshold
PC_RED_THRESHOLD = 1e-4         # Emergency alert threshold
PC_AMBER_THRESHOLD = 1e-5       # Watch alert threshold
```

## UI Dashboard

The operator dashboard includes:

1. **Tracking Summary** - Total objects tracked, RED/AMBER alert counts, closest approach distance, maximum Pc value
2. **Maneuver Decision Panel** - Current decision status (GO/STANDBY/NO GO), target object identification, time to TCA countdown, action recommendations
3. **Propulsion Recommendation** - Recommended option (A/B), suitable thruster list with specifications, estimated delta-V requirement
4. **Active Conjunctions Table** - Object ID, miss distance, probability of collision, time to TCA, risk level
5. **GO/NO-GO Reference Matrix** - Quick reference for decision criteria

## API Reference

### GET `/api/conjunction-status`

Returns current conjunction assessment with decision data.

```json
{
  "status": "active",
  "timestamp": "2024-01-15T12:00:00Z",
  "summary": {
    "total_tracked": 5,
    "red_alerts": 1,
    "amber_alerts": 2,
    "highest_pc": 2.5e-4,
    "closest_approach_km": 0.015
  },
  "primary_recommendation": {
    "object": "DEB_001",
    "decision": "GO",
    "urgency": "CRITICAL",
    "action": "EXECUTE COLLISION AVOIDANCE",
    "time_remaining": "25.3 min",
    "propulsion_option": "A"
  },
  "conjunctions": [...]
}
```

### GET `/api/decision-matrix`

Returns the GO/NO-GO decision matrix configuration.

### POST `/predict` (Detector Service)

Submit SWIR image for debris detection.

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@swir_frame.png"
```

Response:
```json
{
  "frame_id": "frame_001",
  "timestamp_utc": "2024-01-15T12:00:00Z",
  "detections": [
    {
      "class": "debris",
      "confidence": 0.95,
      "bbox": [120, 340, 180, 400],
      "estimated_range_km": 50.2
    }
  ]
}
```

## Local Development

To run services locally without Docker:

```bash
# Create shared data directory
mkdir -p /tmp/avera_data

# Set environment variable
export DATA_DIR=/tmp/avera_data

# Run propagator
cd planner-stack/propagate
pip install -r requirements.txt
python propagator_kepler.py
```

## Kubernetes Deployment

For production deployment on K3s or GKE:

```bash
# Apply namespace
kubectl apply -f k8s/00-namespace.yaml

# Deploy services
kubectl apply -f k8s/01-detector.yaml
kubectl apply -f k8s/02-planner.yaml
kubectl apply -f k8s/03-viz.yaml
kubectl apply -f k8s/04-ui.yaml

# Check status
kubectl get pods -n avera-atlas
```

## References

- Sampson, K. "Thoughts on MMOD Risk & Propulsion System Sizing"
- Alfano, S. (2005). "A Numerical Implementation of Spherical Object Collision Probability"
- NASA Conjunction Assessment Risk Analysis (CARA) Operations
- Frisbee, R. "Probability of Collision for Close Approaches"
- Garrett Reisman's USC Course: Engineering Principles for Human Spaceflight

## Future Enhancements

1. **Automated Maneuver Planning** - Generate optimal burn profiles
2. **Monte Carlo Analysis** - Uncertainty propagation for decision confidence
3. **Historical Trend Analysis** - Pc evolution over time
4. **Multi-Object Coordination** - Prioritized maneuver sequencing
5. **Ground Integration** - Mission Control notification system

## License

Proprietary - xOrbita Inc.
