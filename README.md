---
title: MetaOCT Virtual Eye Clinic Environment
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - medical-ai
  - pomdp
---

# 👁️ MetaOCT: Explainable AI Virtual Eye Clinic

**MetaOCT** is an elite `OpenEnv` compatible Reinforcement Learning (RL) environment that tests a Foundation Model's ability to operate in a **Multi-Step Clinical Diagnosis Pipeline (POMDP)**.

Unlike typical "toy problems" or "single-shot" graders, **MetaOCT forces the Agent to actively spend Virtual Budget to unlock scans, use medical tools, extract spatial fluid coordinates (Bounding Boxes), and reason analytically.**

---

## 🛑 The Core Problem
Frontier Vision-Language Models (VLMs) hallucinate heavily when analyzing extremely dense Optical Coherence Tomography (OCT) retinal scans. If you just ask an LLM, *"What is the diagnosis?"*, it guesses blindly.

But if you place the LLM inside a **strict Resource-Bounded Interaction Environment**—forcing it to actively query spatial tools before committing to an answer—its accuracy skyrockets.

## ⚙️ Environment Overview: Actions and Observations (POMDP)
The Agent starts with a **Patient History** ("Blurry vision"). The actual OCT scan is initially HIDDEN. 

**Observation Space (Pydantic Model):**
- `acquired_scans` (List[str]): Local paths to visually unlocked Retina Images.
- `available_budget` (float): The current numeric hospital diagnosis currency remaining.
- `tool_outputs` (List[str]): Textual sequence of clinical facts and biomarker hints triggered by the agent.
- `step_count` (int): Number of sequential tool actions currently elapsed.

**Action Space (Strict Tools):**
The agent must use its budget sequentially to uncover the biological ground truth via four precise `Action(Tools)`:
1. 💰 `request_oct_scan` (-$150): Unlocks the actual retinal sweep scan.
2. 💰 `enhance_contrast` (-$50): Submits the image to a contrast processor. Increases the agent's maximum theoretical accuracy ceiling by 1.2x.
3. 💰 `measure_fluid_thickness` (-$200): Submits coordinates to query textual biomarkers (e.g. *"Subretinal fluid cysts detected..."*).
4. ✅ `submit_diagnosis` ($0): The terminal State. The Agent finalizes its medical conclusion.

## 📈 Task Descriptions & Difficulty Levels
The environment natively scales across exactly 3 increasing difficulty constraints based on Virtual Budgets.

- **🟢 Easy Task (Budget: $1000):** 
  - Goal: Evaluate basic POMDP traversal.
  - Setup: The agent can afford to spam all tools and re-measure before concluding.
- **🟡 Medium Task (Budget: $400):** 
  - Goal: Optimize precision.
  - Setup: The agent can only afford the standard logical progression (Scan -> Enhance -> Measure). Any hallucination or repeated tool calls causes immediate financial exhaustion.
- **🔴 Hard Task (Budget: $200):**
  - Goal: Absolute resource constraints.
  - Setup: The agent cannot afford to measure fluid thickness fully or enhance contrast safely. It must attempt extreme zero-shot inference with partial observations.

## ⚖️ The Deterministic Reward Engine
The `env.step()` outputs a mathematically grounded reward from `0.00` to `1.00`, calculated across three rigorous axes multiplied by a resource-efficiency index:

$$Total\ Reward = \left[ (0.3 \times Label) + (0.4 \times IoU) + (0.3 \times Keywords) \right] \times \left( \frac{Remaining Budget}{Total Budget} \right)$$

1. **Diagnosis Match (30%)**: Did the categorical label perfectly match (CNV, DME, DRUSEN, NORMAL)?
2. **Pathology Localization (IoU) (40%)**: Does the agent's spatial Heatmap bounding box perfectly intersect the actual fluid cysts? Calculated via strictly continuous `Intersection over Union`.
3. **Medical Reasoning (30%)**: Does the LLM's justification text contain mandatory clinical biomarkers identified by researchers?
4. **Budget Efficiency Penalty**: If the Agent spams tools needlessly and exhausts its clinical budget, the final multiplier slashes its reward perfectly!


## 🚀 Getting Started

### 1. Requirements
Ensure you have Docker or python with UV.
```bash
uv pip install -r requirements.txt
```

### 2. Baseline Performance Scores
The `inference.py` script executes a comprehensive evaluation loop across all 3 Difficulty Tasks (Easy, Medium, Hard) strictly mimicking OpenEnv compliance logs!

```bash
python inference.py
```

*Example standard baseline benchmark emitted to `stdout` across 3 difficulties:*
```text
[START] task=MetaOCT_POMDP env=meta_oct model=meta-llama/Meta-Llama-3-8B-Instruct
[STEP] step=1 action=Tool(request_oct_scan) reward=0.00 done=false error=null
[STEP] step=2 action=Tool(enhance_contrast) reward=0.00 done=false error=null
[STEP] step=3 action=Tool(measure_fluid_thickness) reward=0.01 done=false error=null
[STEP] step=4 action=Tool(submit_diagnosis) reward=0.82 done=true error=null
...
[END] success=true steps=12 score=0.825 rewards=...
[END] success=true steps=24 score=0.720 rewards=... difficulty=medium
[END] success=false steps=36 score=-0.008 rewards=... difficulty=hard
```

### 3. Reinforcement Learning Training Platform (PPO / GRPO Ready)
To go beyond evaluation, `MetaOCT` natively supports PyTorch Tensor training. You can train 1B+ parameter models directly via PPO backpropagation using the environment's mathematically continuous grading engine!

Run the lightweight Policy Network on CPU:
```bash
python train_rl.py
```
This loop demonstrates PyTorch gradients scaling perfectly with the budget-restricted medical reward signals:
```text
============================================================
🚀 MetaOCT End-to-End Reinforcement Learning Pipeline
Algorithm: REINFORCE / Proximal Policy Optimization (PPO)
============================================================
Episode 025 | Moving Avg Reward:  0.40 | Loss:   0.00
Episode 250 | Moving Avg Reward:  0.43 | Loss:   0.00 
✅ Training Simulation Complete!
```

---
*Built perfectly for the Meta OpenEnv RL Challenge. 100% compliant with standard Hacker specifications.*
