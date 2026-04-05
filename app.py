"""
MetaOCT: OpenEnv-Compliant REST API Server
Implements POST /reset, POST /step, GET /validate per OpenEnv specification.
Hackathon automated checker compatible.
"""

import json
import os
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
from env import MetaOCTEnv, Action

load_dotenv()

# Global environment instance and state
_env = None
_env_lock = threading.Lock()
_loop = None

def get_loop():
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop

def run_async(coro):
    loop = get_loop()
    return loop.run_until_complete(coro)

def get_env(difficulty="easy"):
    global _env
    if _env is None:
        _env = MetaOCTEnv(difficulty=difficulty)
    return _env

class OpenEnvHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass  # Suppress access logs

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            return json.loads(self.rfile.read(length))
        return {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/validate" or self.path == "/openenv/validate":
            self.send_json(200, {
                "status": "ok",
                "name": "MetaOCT-Simulator",
                "version": "1.0.0",
                "tasks": ["easy", "medium", "hard"],
                "action_space": ["request_oct_scan", "enhance_contrast", "measure_fluid_thickness", "submit_diagnosis"],
                "observation_space": ["acquired_scans", "available_budget", "tool_outputs", "step_count"]
            })
        elif self.path == "/" or self.path == "/health":
            self.send_json(200, {"status": "running", "env": "MetaOCT-Simulator"})
        elif self.path == "/openenv/info":
            self.send_json(200, {
                "name": "MetaOCT-Simulator",
                "description": "Multi-step POMDP clinical OCT diagnostic environment",
                "tasks": [
                    {"id": "easy", "budget": 1000, "description": "Basic POMDP traversal"},
                    {"id": "medium", "budget": 400, "description": "Precision optimization"},
                    {"id": "hard", "budget": 200, "description": "Extreme resource constraints"}
                ]
            })
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        try:
            body = self.read_body()

            if self.path in ["/reset", "/openenv/reset"]:
                difficulty = body.get("task", body.get("difficulty", "easy"))
                with _env_lock:
                    global _env
                    _env = MetaOCTEnv(difficulty=difficulty)
                    obs = run_async(_env.reset())
                
                self.send_json(200, {
                    "observation": {
                        "acquired_scans": obs.acquired_scans,
                        "available_budget": obs.available_budget,
                        "tool_outputs": obs.tool_outputs,
                        "step_count": obs.step_count
                    },
                    "info": {"difficulty": difficulty, "task": difficulty}
                })

            elif self.path in ["/step", "/openenv/step"]:
                action_data = body.get("action", body)
                tool_name = action_data.get("tool_name", action_data.get("name", "submit_diagnosis"))
                parameters = action_data.get("parameters", action_data.get("args", {}))
                action = Action(tool_name=tool_name, parameters=parameters)
                
                with _env_lock:
                    env = get_env()
                    result = run_async(env.step(action))
                
                obs = result.observation
                obs_dict = {}
                if obs:
                    obs_dict = {
                        "acquired_scans": obs.acquired_scans,
                        "available_budget": obs.available_budget,
                        "tool_outputs": obs.tool_outputs,
                        "step_count": obs.step_count
                    }
                
                self.send_json(200, {
                    "observation": obs_dict,
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info or {}
                })

            else:
                self.send_json(404, {"error": "Unknown endpoint"})

        except Exception as e:
            print(f"[ERROR] {self.path}: {e}", flush=True)
            self.send_json(500, {"error": str(e)})


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    # Start HTTP server IMMEDIATELY so checker can connect
    server = HTTPServer(("0.0.0.0", port), OpenEnvHandler)
    print(f"[INFO] MetaOCT OpenEnv API running on port {port}", flush=True)
    print(f"[INFO] Endpoints: POST /reset, POST /step, GET /validate", flush=True)
    
    # Pre-warm environment in background thread
    def prewarm():
        print("[INFO] Pre-warming MetaOCT environment...", flush=True)
        get_env("easy")
        print("[INFO] Environment ready.", flush=True)
    
    t = threading.Thread(target=prewarm, daemon=True)
    t.start()
    
    server.serve_forever()
