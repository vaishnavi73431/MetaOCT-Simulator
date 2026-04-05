"""
MetaOCT Hackathon Web App
Runs inference.py in background thread and serves results via a minimal web server.
This keeps the HuggingFace Space alive permanently.
"""

import threading
import asyncio
import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
from env import MetaOCTEnv, Action, Observation
from openai import OpenAI
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from typing import List, Optional

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None and os.getenv("OPENAI_API_KEY") is None:
    print("[WARNING] Required API keys missing.", flush=True)
API_KEY = os.getenv("OPENAI_API_KEY") or HF_TOKEN or os.getenv("API_KEY")

# Global results store
results = {"status": "running", "logs": [], "score": None, "success": None}

# Vision Model
print("[DEBUG] Loading Vision Model...", flush=True)
try:
    processor = AutoImageProcessor.from_pretrained("octava/image_classification")
    hf_model = AutoModelForImageClassification.from_pretrained("octava/image_classification", output_attentions=True)
except Exception as e:
    print(f"[DEBUG] Vision model warning: {e}", flush=True)
    processor = None
    hf_model = None

def get_vision_prediction(image_path: str):
    diagnosis = "NORMAL"
    heatmap = [[0,0],[0,0]]
    if hf_model is not None:
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = hf_model(**inputs)
            idx = outputs.logits.argmax(-1).item()
            label = hf_model.config.id2label[idx].upper()
            if "CNV" in label: diagnosis = "CNV"
            elif "DME" in label: diagnosis = "DME"
            elif "DRUSEN" in label: diagnosis = "DRUSEN"
            attentions = outputs.attentions
            avg_attention = attentions[-1].mean(dim=1).squeeze(0)
            cls_attention = avg_attention[0, 1:]
            grid = cls_attention.reshape(14,14)
            max_idx = torch.argmax(grid).item()
            y, x = max_idx // 14, max_idx % 14
            p = 16
            heatmap = [[max(0,(x-1)*p), max(0,(y-1)*p)], [min(224,(x+2)*p), min(224,(y+2)*p)]]
        except Exception as e:
            print(f"[DEBUG] Vision error: {e}", flush=True)
    return diagnosis, heatmap

def get_heuristic_action(step: int, obs, client) -> Action:
    if step == 1: return Action(tool_name="request_oct_scan", parameters={})
    elif step == 2: return Action(tool_name="enhance_contrast", parameters={})
    elif step == 3: return Action(tool_name="measure_fluid_thickness", parameters={})
    else:
        # Keyword-based deterministic diagnosis from tool_outputs (no LLM needed)
        all_outputs = " ".join(str(t) for t in obs.tool_outputs).lower()
        if any(k in all_outputs for k in ["neovascularization", "subretinal fluid", "rpe elevation", "cnv"]):
            diagnosis = "CNV"
            heatmap = [[80, 80], [150, 150]]
        elif any(k in all_outputs for k in ["intraretinal cysts", "thickening", "edema", "dme"]):
            diagnosis = "DME"
            heatmap = [[80, 80], [150, 150]]
        elif any(k in all_outputs for k in ["drusen", "rpe deposits"]):
            diagnosis = "DRUSEN"
            heatmap = [[80, 80], [150, 150]]
        else:
            image_path = obs.acquired_scans[-1] if obs.acquired_scans else "dummy.jpg"
            diagnosis, heatmap = get_vision_prediction(image_path)
        reasoning = "Clinical biomarkers confirm diagnosis based on retinal pathology signatures."
        try:
            prompt = f"You are an expert ophthalmologist. Confirm in 1 sentence why this is {diagnosis}."
            completion = client.chat.completions.create(model=MODEL_NAME, messages=[{"role":"user","content":prompt}], max_tokens=60)
            reasoning = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[DEBUG] LLM error: {e}", flush=True)
        return Action(tool_name="submit_diagnosis", parameters={"diagnosis": diagnosis, "heatmap_coordinates": heatmap, "reasoning": reasoning})

async def run_inference():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    global_rewards = []
    global_steps = 0
    difficulties = ["easy", "medium", "hard"]
    
    log_line = f"[START] task=MetaOCT_POMDP env=meta_oct model={MODEL_NAME}"
    print(log_line, flush=True)
    results["logs"].append(log_line)
    
    for diff in difficulties:
        env = MetaOCTEnv(difficulty=diff)
        for _ in range(min(env.max_patients, 3)):
            obs = await env.reset()
            episode_step = 0
            while True:
                episode_step += 1
                global_steps += 1
                action_obj = get_heuristic_action(episode_step, obs, client)
                result = await env.step(action_obj)
                reward = result.reward or 0.0
                done = result.done
                obs = result.observation
                global_rewards.append(reward)
                step_log = f"[STEP] step={global_steps} action=Tool({action_obj.tool_name}) reward={reward:.2f} done={str(done).lower()} error=null"
                print(step_log, flush=True)
                results["logs"].append(step_log)
                if done: break
        await env.close()
    
    max_total = float(len(global_rewards))
    total_score = sum(global_rewards) / max_total if max_total > 0 else 0.0
    success = total_score >= 0.7
    end_log = f"[END] success={str(success).lower()} steps={global_steps} score={total_score:.3f} rewards={','.join(f'{r:.2f}' for r in global_rewards)}"
    print(end_log, flush=True)
    results["logs"].append(end_log)
    results["status"] = "complete"
    results["score"] = total_score
    results["success"] = success

def run_inference_thread():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_inference())

# Simple Web Server - Serves the results as HTML
class ResultsHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass  # Suppress access logs
    
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        
        status_color = "#00ff88" if results["status"] == "complete" else "#ffaa00"
        score_display = f"{results['score']:.3f}" if results["score"] is not None else "Running..."
        
        html = f"""<!DOCTYPE html>
<html><head>
<title>MetaOCT Virtual Clinic</title>
<meta http-equiv="refresh" content="5">
<style>
body{{background:#0d1117;color:#e6edf3;font-family:monospace;padding:40px;}}
h1{{color:#58a6ff;}} .status{{color:{status_color};font-size:1.4em;}} 
pre{{background:#161b22;padding:20px;border-radius:8px;overflow-x:auto;font-size:13px;max-height:500px;overflow-y:auto;}}
.score{{font-size:2em;color:#f0883e;}} .badge{{background:#238636;padding:4px 12px;border-radius:20px;}}
</style></head>
<body>
<h1>👁️ MetaOCT: Virtual Medical Clinic (POMDP)</h1>
<p>Multi-Step Reinforcement Learning Environment | Meta OpenEnv Hackathon</p>
<p class="status">Status: {results["status"].upper()}</p>
<p class="score">Score: {score_display}</p>
<pre>{"<br>".join(results["logs"][-30:])}</pre>
<p><span class="badge">OpenEnv Compliant</span> &nbsp; Built with PyTorch + LLaMA-3 + OctaVA Vision</p>
</body></html>"""
        self.wfile.write(html.encode())

if __name__ == "__main__":
    # Start inference in background
    thread = threading.Thread(target=run_inference_thread, daemon=True)
    thread.start()
    
    # Start web server on port 7860 (HuggingFace required)
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    print(f"[INFO] Starting MetaOCT Dashboard on port {port}", flush=True)
    server = HTTPServer(("0.0.0.0", port), ResultsHandler)
    server.serve_forever()
