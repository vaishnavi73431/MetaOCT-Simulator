import json
import logging
import os
from typing import Literal, List, Optional, Dict, Any
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Observation(BaseModel):
    clinical_notes: str
    available_budget: int
    acquired_scans: List[str]
    tool_outputs: List[str]
    step_count: int
    task_id: Literal["easy", "medium", "hard"]

class Action(BaseModel):
    tool_name: Literal["request_oct_scan", "enhance_contrast", "measure_fluid_thickness", "submit_diagnosis"]
    parameters: Dict[str, Any]

class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool
    info: dict

def calculate_iou(box1: List[List[int]], box2: List[List[int]]) -> float:
    x1_inter = max(box1[0][0], box2[0][0])
    y1_inter = max(box1[0][1], box2[0][1])
    x2_inter = min(box1[1][0], box2[1][0])
    y2_inter = min(box1[1][1], box2[1][1])

    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[1][0] - box1[0][0]) * (box1[1][1] - box1[0][1])
    box2_area = (box2[1][0] - box2[0][0]) * (box2[1][1] - box2[0][1])
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

class MetaOCTEnv:
    def __init__(self, data_dir: str = ".", truth_file: str = "ground_truth.json", difficulty: str = "medium"):
        self.data_dir = data_dir
        with open(truth_file, "r") as f:
            self.ground_truth = json.load(f)
        self.image_files = list(self.ground_truth.keys())
        self.current_idx = 0
        self.max_patients = len(self.image_files)
        
        self.difficulty = difficulty.lower()
        if self.difficulty == "easy":
            self.initial_budget = 1000
        elif self.difficulty == "hard":
            self.initial_budget = 200
        else:
            self.initial_budget = 400
            
        self.available_budget = self.initial_budget
        self.acquired_scans = []
        self.tool_outputs = []
        self.step_count = 0
        self.max_steps = 10
        self.contrast_enhanced = False

    def state(self) -> dict:
        return {
            "current_idx": self.current_idx,
            "max_patients": self.max_patients,
            "is_done": self.current_idx >= self.max_patients
        }

    async def reset(self) -> Observation:
        self.available_budget = self.initial_budget
        self.acquired_scans = []
        self.tool_outputs = [f"Patient arrived. You have a ${self.initial_budget} diagnostic budget."]
        self.step_count = 0
        self.contrast_enhanced = False
        return self._get_observation()

    def _get_observation(self) -> Observation:
        img_name = self.image_files[self.current_idx % len(self.image_files)]
        truth = self.ground_truth[img_name]
        
        task_id = "easy"
        if "CNV" in truth["label"]: task_id = "hard"
        elif "DME" in truth["label"] or "DRUSEN" in truth["label"]: task_id = "medium"

        clinical_notes = "Patient complains of blurry vision."
        if task_id == "easy": clinical_notes = "Routine yearly diabetic eye checkup."
        
        return Observation(
            clinical_notes=clinical_notes,
            available_budget=self.available_budget,
            acquired_scans=self.acquired_scans,
            tool_outputs=self.tool_outputs[-5:], # Keep last 5 outputs to prevent context bloat
            step_count=self.step_count,
            task_id=task_id
        )

    async def step(self, action: Action) -> StepResult:
        if self.current_idx >= self.max_patients:
            return StepResult(observation=None, reward=0.0, done=True, info={})

        self.step_count += 1
        img_name = self.image_files[self.current_idx]
        truth = self.ground_truth[img_name]
        
        reward = 0.0
        done = False
        info = {}
        
        if self.step_count >= self.max_steps and action.tool_name != "submit_diagnosis":
            done = True
            self.current_idx += 1
            info = {"error": "Max steps reached before diagnosis"}
            return StepResult(observation=None, reward=-1.0, done=done, info=info)

        if action.tool_name == "request_oct_scan":
            cost = 150
            if self.available_budget >= cost:
                self.available_budget -= cost
                img_path = os.path.join(self.data_dir, img_name)
                if img_path not in self.acquired_scans:
                    self.acquired_scans.append(img_path)
                    self.tool_outputs.append(f"[request_oct_scan] Success. Scan acquired at {img_path}.")
                else:
                    reward -= 0.05
                    self.tool_outputs.append("[request_oct_scan] Warning: Scan already acquired. Wasted budget.")
            else:
                reward -= 0.1
                self.tool_outputs.append("[request_oct_scan] Error: Insufficient budget.")

        elif action.tool_name == "enhance_contrast":
            cost = 50
            if self.available_budget >= cost:
                self.available_budget -= cost
                if not self.acquired_scans:
                    reward -= 0.05
                    self.tool_outputs.append("[enhance_contrast] Error: No scan to enhance. Request scan first.")
                elif self.contrast_enhanced:
                    reward -= 0.05
                    self.tool_outputs.append("[enhance_contrast] Warning: Already enhanced. Wasted budget.")
                else:
                    self.contrast_enhanced = True
                    self.tool_outputs.append("[enhance_contrast] Success. Vision clarity improved by 1.2x.")
            else:
                reward -= 0.1
                self.tool_outputs.append("[enhance_contrast] Error: Insufficient budget.")

        elif action.tool_name == "measure_fluid_thickness":
            cost = 200
            if self.available_budget >= cost:
                self.available_budget -= cost
                if not self.acquired_scans:
                    reward -= 0.05
                    self.tool_outputs.append("[measure_fluid] Error: No scan to measure. Request scan first.")
                else:
                    if truth["label"] in ["CNV", "DME"]:
                        msg = f"[measure_fluid] Abnormal retinal thickening detected. Biomarkers found: {', '.join(truth['keywords'])}"
                    else:
                        msg = "[measure_fluid] Normal foveal contour observed. No abnormal fluid."
                    self.tool_outputs.append(msg)
            else:
                reward -= 0.1
                self.tool_outputs.append("[measure_fluid] Error: Insufficient budget.")

        elif action.tool_name == "submit_diagnosis":
            done = True
            
            diagnosis = action.parameters.get("diagnosis", "")
            heatmap = action.parameters.get("heatmap_coordinates", [[0,0],[0,0]])
            reasoning = action.parameters.get("reasoning", "")
            
            label_match = 1.0 if diagnosis.upper() == truth["label"].upper() else 0.0
            
            true_box = truth["box"]
            iou_score = 0.0
            if len(heatmap) >= 2 and len(heatmap[0]) >= 2 and len(heatmap[1]) >= 2:
                iou_score = calculate_iou(heatmap, true_box)
                if true_box[0] == [0,0] and true_box[1] == [0,0]:
                    iou_score = 1.0 if (heatmap[0] == [0,0] and heatmap[1] == [0,0]) else 0.0
                    
            if self.contrast_enhanced:
                iou_score = min(1.0, iou_score * 1.2)
                
            reasoning_lower = reasoning.lower()
            if truth["keywords"]:
                matched = sum(1 for kw in truth["keywords"] if kw.lower() in reasoning_lower)
                reasoning_score = matched / len(truth["keywords"])
            else:
                reasoning_score = 1.0
                
            base_reward = (0.3 * label_match) + (0.4 * iou_score) + (0.3 * reasoning_score)
            budget_efficiency = max(0.2, self.available_budget / self.initial_budget)
            
            reward += (base_reward * budget_efficiency)
            
            info = {
                "label_match": label_match,
                "iou_score": iou_score,
                "reasoning_score": reasoning_score,
                "budget_efficiency": budget_efficiency,
                "true_label": truth["label"],
                "final_base_score": base_reward
            }
            
            self.tool_outputs.append(f"[submit_diagnosis] Evaluated. Score: {reward:.2f}")
            self.current_idx += 1

        else:
            reward -= 0.1
            self.tool_outputs.append(f"[{action.tool_name}] Unknown tool.")
            
        obs = self._get_observation() if not done else None
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    async def close(self):
        pass
