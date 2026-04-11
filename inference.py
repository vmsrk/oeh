#!/usr/bin/env python3
import sys, subprocess, importlib
def ensure_package(import_name, pip_spec):
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"[INFO] Installing {pip_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_spec])
        importlib.import_module(import_name)

ensure_package("openai", "openai>=1.6.0")
ensure_package("pydantic", "pydantic>=2.5.0")
ensure_package("dotenv", "python-dotenv>=1.0.0")

import asyncio, os, json, textwrap
from typing import List, Optional
from openai import OpenAI
from clinical_triage_env import ClinicalTriageEnv
from clinical_triage_env.models import Action, TestType, QuestionType, TriageLevel

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or "ollama"
TASK_NAMES = ["easy", "medium", "hard"]
MAX_STEPS = 10
TEMPERATURE = 0.3

SYSTEM_PROMPT = """You are an emergency triage nurse. Patient deteriorates over time. Order tests, ask questions, then assign triage (0=critical,1=high,2=medium,3=low). Tests cost time and money. Asking relevant questions reveals hidden symptoms. Be efficient – deterioration increases every step. Return ONLY JSON."""

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}", flush=True)

def extract_json(text):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
    return None

def fallback_action(difficulty: str) -> Action:
    if difficulty == "easy":
        return Action(assign_triage=3)
    elif difficulty == "medium":
        return Action(assign_triage=1)
    else:
        return Action(assign_triage=0)

async def get_model_action(client, case, step, history, deterioration, revealed):
    difficulty = case.difficulty
    if step >= 5:
        print(f"[DEBUG] Step {step}: forcing fallback for {difficulty}", flush=True)
        return fallback_action(difficulty)

    hist_str = "\n".join(history[-3:]) if history else "None"
    user_prompt = f"""
Patient: {case.age}y, HR={case.heart_rate}, BP={case.systolic_bp}, SpO2={case.oxygen_saturation}%, T={case.temperature}°C, Pain={case.pain_level}/10
Arrival: {case.arrival_mode}, Chronic: {case.chronic_disease_count}, Prior ER: {case.previous_er_visits}
Chief complaint: {case.chief_complaint}
History: {case.medical_history}
Deterioration: {deterioration:.2f} (0=stable,1=critical)
Revealed symptoms: {revealed}
Step: {step}
History: {hist_str}
Decide next action (order_test, ask_question, or assign_triage). Return ONLY JSON.
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user_prompt}],
            temperature=TEMPERATURE,
        )
        data = extract_json(resp.choices[0].message.content.strip())
        if not data:
            raise ValueError("Invalid JSON")
        if "order_test" in data:
            return Action(order_test=data["order_test"])
        if "ask_question" in data:
            return Action(ask_question=data["ask_question"])
        if "assign_triage" in data:
            return Action(assign_triage=int(data["assign_triage"]))
        return fallback_action(difficulty)
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}, fallback", flush=True)
        return fallback_action(difficulty)

async def run_episode(env, client, task_id):
    history = []
    rewards = []
    obs = await env.reset(task_id)
    step = 1
    done = False
    while not done and step <= MAX_STEPS:
        action = await get_model_action(client, env.current_case, step, history,
                                        obs.deterioration, obs.revealed_symptoms)
        action_str = action.model_dump_json()
        obs, reward, done, info = await env.step(action)
        reward_val = reward.value
        rewards.append(reward_val)
        log_step(step, action_str, reward_val, done, info.get("error"))
        history.append(f"Step {step}: {action_str} -> {reward_val:.2f}")
        step += 1
    state = await env.state()
    final_score = state.final_score if state.final_score is not None else 0.0
    success = final_score >= 0.49
    return {"success": success, "steps": step-1, "score": final_score, "rewards": rewards}

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = ClinicalTriageEnv()
    for task_id in TASK_NAMES:
        log_start(task=task_id, env="clinical_triage_env", model=MODEL_NAME)
        res = await run_episode(env, client, task_id)
        log_end(success=res["success"], steps=res["steps"], score=res["score"], rewards=res["rewards"])
        print()

if __name__ == "__main__":
    asyncio.run(main())