import base64
import requests
import os
import uuid
import json
import numpy as np
from datetime import datetime
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="AVERA-ATLAS", version="3.0.0")

SWIR_SERVICE_URL = os.getenv("SWIR_SERVICE_URL", "http://swir-detector:8000/predict")
TRACKER_SERVICE_URL = os.getenv("TRACKER_SERVICE_URL", "http://tracker:8000")
DATA_DIR = os.getenv("DATA_DIR", "/data/planner_artifacts")
VIDEO_ARTIFACT_PATH = os.path.join(DATA_DIR, "planner_output.mp4")
PROP_ARTIFACT_PATH = os.path.join(DATA_DIR, "prop_multi.npz")

templates = Jinja2Templates(directory="app/templates")


# =============================================================================
# Decision Logic Functions
# =============================================================================

def evaluate_go_nogo(pc: float, time_to_tca_min: float):
    """Evaluate GO/NO GO decision based on Pc and time to TCA."""
    if time_to_tca_min < 30:
        if pc >= 1e-3:
            return {"decision": "GO", "urgency": "CRITICAL", "confidence": 0.95}
        elif pc >= 1e-4:
            return {"decision": "STANDBY", "urgency": "HIGH", "confidence": 0.85}
        else:
            return {"decision": "NO_GO", "urgency": "MONITOR", "confidence": 0.80}
    elif time_to_tca_min < 120:
        if pc >= 1e-4:
            return {"decision": "GO", "urgency": "HIGH", "confidence": 0.90}
        elif pc >= 1e-5:
            return {"decision": "STANDBY", "urgency": "ELEVATED", "confidence": 0.80}
        else:
            return {"decision": "NO_GO", "urgency": "LOW", "confidence": 0.85}
    else:
        if pc >= 1e-5:
            return {"decision": "GO", "urgency": "MODERATE", "confidence": 0.85}
        elif pc >= 1e-6:
            return {"decision": "STANDBY", "urgency": "LOW", "confidence": 0.75}
        else:
            return {"decision": "NO_GO", "urgency": "NOMINAL", "confidence": 0.90}


def get_thruster_recommendation(time_to_tca_min: float, decision: str):
    """Get propulsion system recommendation based on available time."""
    if decision == "NO_GO":
        return {"option": "N/A", "thrusters": [], "rationale": "No maneuver required"}
    
    if time_to_tca_min < 30:
        return {
            "option": "A",
            "option_name": "High Thrust",
            "thrusters": [
                {"name": "Cold Gas", "thrust": "10-1000 mN", "isp": "50-80s", "response": "0.1s"},
                {"name": "Monopropellant", "thrust": "100-5000 mN", "isp": "200-250s", "response": "0.5s"}
            ],
            "rationale": "Limited time requires high-thrust, fast-response propulsion"
        }
    elif time_to_tca_min < 120:
        return {
            "option": "A/B",
            "option_name": "Flexible",
            "thrusters": [
                {"name": "Monopropellant", "thrust": "100-5000 mN", "isp": "200-250s", "response": "0.5s"},
                {"name": "Hall Thruster", "thrust": "50-500 mN", "isp": "1200-2000s", "response": "30s"}
            ],
            "rationale": "Moderate time allows for either option; balance thrust vs efficiency"
        }
    else:
        return {
            "option": "B",
            "option_name": "High Efficiency",
            "thrusters": [
                {"name": "Electrospray", "thrust": "0.01-1 mN", "isp": "500-2500s", "response": "5s"},
                {"name": "FEEP", "thrust": "0.001-0.5 mN", "isp": "4000-10000s", "response": "10s"},
                {"name": "Mini Ion", "thrust": "0.1-5 mN", "isp": "2000-4000s", "response": "60s"}
            ],
            "rationale": "Ample time allows for efficient, low-thrust maneuvers"
        }


def format_time_display(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds/3600:.1f} hr"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/video")
async def get_video():
    if os.path.exists(VIDEO_ARTIFACT_PATH):
        return FileResponse(VIDEO_ARTIFACT_PATH, media_type="video/mp4", headers={"Cache-Control": "no-cache"})
    else:
        return {"error": "Trajectory visualization not ready."}


@app.get("/api/summary")
async def get_summary():
    summary_path = os.path.join(DATA_DIR, "conjunction_summary.png")
    if os.path.exists(summary_path):
        return FileResponse(summary_path, media_type="image/png", headers={"Cache-Control": "no-cache"})
    else:
        return {"error": "Summary chart not ready."}


@app.post("/api/process")
async def process_image(
        file: UploadFile = File(None),
        text_input: str = Form(None)
):
    img_b64 = ""
    if file:
        file_content = await file.read()
        img_b64 = base64.b64encode(file_content).decode('utf-8')
    elif text_input:
        img_b64 = text_input.split(",")[1] if "," in text_input else text_input
    else:
        return {"error": "No image provided"}

    try:
        # Construct Payload
        payload = {
            "frame_id": f"ui_scan_{uuid.uuid4().hex[:8]}",
            "base64_data": img_b64,
            "sensor_id": "ui_test_sensor_01",
            "camera_pose": {
                "position_eci_km": [6871.0, -1200.5, 300.2],
                "quaternion_eci_body": [0.707, 0.0, -0.707, 0.0]
            }
        }

        response = requests.post(SWIR_SERVICE_URL, json=payload, timeout=10)
        if response.status_code != 200:
            return {"error": f"Detector Error: {response.text}"}

        data = response.json()

        # Format for UI
        ui_detections = []
        for det in data.get("detections", []):
            raw_bbox = det.get("bbox", [0, 0, 0, 0])
            ui_detections.append({
                "box": raw_bbox,
                # Use specific spacecraft type if available, fallback to class
                "label": det.get("spacecraft_type") or det.get("class", "unknown"),
                "class": det.get("class", "unknown"),  # Keep category for styling/logic
                "confidence": det.get("confidence", 0.0)
            })

        return {
            "detections": ui_detections,
            "metadata": {
                "frame_id": data.get("frame_id"),
                "object_count": len(ui_detections)
            }
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/conjunction-status")
async def get_conjunction_status():
    """Get current conjunction assessment status with decision data."""
    if not os.path.exists(PROP_ARTIFACT_PATH):
        return JSONResponse({
            "status": "no_data",
            "message": "No conjunction data available. Run a detection scan first.",
            "conjunctions": []
        })
    
    try:
        data = np.load(PROP_ARTIFACT_PATH, allow_pickle=True)
        
        obj_ids = data['obj_ids']
        ca_table = data['ca_table']  # Miss distances (km)
        pc_values = data.get('pc_values', np.zeros(len(obj_ids)))
        risk_levels = data.get('risk_levels', np.array(['NOMINAL'] * len(obj_ids)))
        tca_indices = data.get('tca_indices', np.zeros(len(obj_ids), dtype=int))
        rel_velocities = data.get('relative_velocities', np.zeros(len(obj_ids)))
        
        n_red = int(data.get('n_red_alerts', 0))
        n_amber = int(data.get('n_amber_alerts', 0))
        
        # Assume 60s time step
        dt_sec = 60.0
        
        conjunctions = []
        for i in range(len(obj_ids)):
            time_to_tca_s = float(tca_indices[i]) * dt_sec
            time_to_tca_min = time_to_tca_s / 60.0
            
            pc = float(pc_values[i])
            miss_km = float(ca_table[i])
            rel_vel = float(rel_velocities[i])
            risk = str(risk_levels[i])
            
            decision_result = evaluate_go_nogo(pc, time_to_tca_min)
            propulsion = get_thruster_recommendation(time_to_tca_min, decision_result['decision'])
            
            # Simple delta-V estimate
            delta_v = 0.0
            if miss_km < 1.0 and time_to_tca_s > 0:
                delta_v = (1.0 - miss_km) / time_to_tca_s * 1000 * 1.5
            
            conjunctions.append({
                "object_id": str(obj_ids[i]),
                "miss_distance_km": round(miss_km, 3),
                "miss_distance_m": round(miss_km * 1000, 1),
                "pc": pc,
                "pc_display": f"{pc:.2e}" if pc > 0 else "N/A",
                "risk_level": risk,
                "time_to_tca_min": round(time_to_tca_min, 1),
                "time_to_tca_display": format_time_display(time_to_tca_s),
                "relative_velocity_km_s": round(rel_vel, 2),
                "decision": decision_result['decision'],
                "decision_urgency": decision_result['urgency'],
                "decision_confidence": decision_result['confidence'],
                "propulsion": propulsion,
                "delta_v_estimate_m_s": round(delta_v, 3) if delta_v > 0 else None
            })
        
        # Sort by risk
        risk_order = {"RED": 0, "AMBER": 1, "GREEN": 2, "NOMINAL": 3}
        conjunctions.sort(key=lambda x: (risk_order.get(x['risk_level'], 4), -x['pc']))
        
        # Get primary recommendation
        primary_recommendation = None
        for conj in conjunctions:
            if conj['decision'] in ['GO', 'STANDBY']:
                primary_recommendation = {
                    "object": conj['object_id'],
                    "decision": conj['decision'],
                    "urgency": conj['decision_urgency'],
                    "action": "EXECUTE COLLISION AVOIDANCE" if conj['decision'] == 'GO' else "PREPARE FOR MANEUVER",
                    "time_remaining": conj['time_to_tca_display'],
                    "propulsion_option": conj['propulsion']['option']
                }
                break
        
        return JSONResponse({
            "status": "active",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "summary": {
                "total_tracked": len(obj_ids),
                "red_alerts": n_red,
                "amber_alerts": n_amber,
                "green_alerts": len([r for r in risk_levels if r == 'GREEN']),
                "highest_pc": float(max(pc_values)) if len(pc_values) > 0 else 0,
                "closest_approach_km": float(min(ca_table)) if len(ca_table) > 0 else None
            },
            "primary_recommendation": primary_recommendation,
            "conjunctions": conjunctions
        })
        
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e),
            "conjunctions": []
        }, status_code=500)


@app.get("/api/tracker-status")
async def get_tracker_status():
    """Get current tracker service status including UCTs and tracks."""
    try:
        # Get tracker status
        status_resp = requests.get(f"{TRACKER_SERVICE_URL}/status", timeout=5)
        if status_resp.status_code != 200:
            return JSONResponse({
                "status": "offline",
                "message": "Tracker service unavailable"
            })
        
        status_data = status_resp.json()
        
        # Get UCT details
        uct_resp = requests.get(f"{TRACKER_SERVICE_URL}/uncorrelated", timeout=5)
        uct_data = uct_resp.json() if uct_resp.status_code == 200 else {}
        ucts = uct_data.get("buffers", [])[:5] if isinstance(uct_data, dict) else []
        
        # Get tracks
        tracks_resp = requests.get(f"{TRACKER_SERVICE_URL}/tracks", timeout=5)
        tracks_data = tracks_resp.json() if tracks_resp.status_code == 200 else {"tracks": []}
        tracks = tracks_data.get("tracks", [])[:5] if isinstance(tracks_data, dict) else []
        
        return JSONResponse({
            "status": "online",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": {
                "version": status_data.get("version", "unknown"),
                "uptime_seconds": status_data.get("uptime_seconds", 0),
                "detections_processed": status_data.get("detections_processed", 0)
            },
            "pipeline": {
                "registered_sensors": status_data.get("registered_sensors", 0),
                "uncorrelated_detections": status_data.get("uncorrelated_detections", 0),
                "active_tracks": status_data.get("active_tracks", 0),
                "confirmed_tracks": status_data.get("confirmed_tracks", 0),
                "tentative_tracks": status_data.get("tentative_tracks", 0)
            },
            "ucts": ucts,
            "tracks": tracks
        })
        
    except requests.exceptions.RequestException:
        return JSONResponse({
            "status": "offline",
            "message": "Cannot connect to tracker service"
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


# Demo mode state
demo_running = False
demo_task = None


@app.post("/api/demo/start")
async def start_demo():
    """Start autonomous demo mode - generates synthetic detections."""
    global demo_running
    
    if demo_running:
        return {"status": "already_running", "message": "Demo is already running"}
    
    demo_running = True
    
    # Trigger initial detection batch via tracker's demo endpoint
    try:
        resp = requests.post(f"{TRACKER_SERVICE_URL}/demo/generate", 
                           json={"scenario": "conjunction", "count": 4},
                           timeout=10)
        if resp.status_code == 200:
            return {
                "status": "started",
                "message": "Demo mode active - generating synthetic detections",
                "scenario": "conjunction"
            }
        else:
            demo_running = False
            return {"status": "error", "message": f"Tracker error: {resp.text}"}
    except Exception as e:
        demo_running = False
        return {"status": "error", "message": str(e)}


@app.post("/api/demo/stop")
async def stop_demo():
    """Stop demo mode."""
    global demo_running
    demo_running = False
    return {"status": "stopped", "message": "Demo mode stopped"}


@app.get("/api/demo/status")
async def demo_status():
    """Get demo mode status."""
    return {"running": demo_running}