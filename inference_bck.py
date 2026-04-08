#!/usr/bin/env python3
import asyncio
import os
import json
import textwrap
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Import from the installed package
from email_triage_env import EmailTriageEnv
from email_triage_env.models import Action, InfoType, Priority, ActionType

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set OPENAI_API_KEY")

TASK_NAMES = ["easy", "medium", "hard"]
MAX_STEPS = 10
TEMPERATURE = 0.3

SYSTEM_PROMPT = textwrap.dedent("""
You are an AI customer support agent. Triage incoming emails.
Options:
- Request info: {"request_info": "sender_history"} (or product_details/previous_tickets)
- Submit: {"submit_priority": "High", "submit_action": "Reply"}
Priorities: High, Medium, Low. Actions: Reply, Archive, Delete, Flag.
Output ONLY the JSON object.
""").strip()

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rstr = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rstr}", flush=True)

def get_model_action(client, task_id, email_text, sender, subject, step, history):
    hist_str = "\n".join(history[-3:]) if history else "None"
    user_prompt = textwrap.dedent(f"""
    Task: {task_id}
    From: {sender}
    Subject: {subject}
    Body:
    {email_text}
    Step: {step}
    History: {hist_str}
    """).strip()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        if "request_info" in data:
            return Action(request_info=InfoType(data["request_info"]))
        elif "submit_priority" in data and "submit_action" in data:
            return Action(
                submit_priority=Priority(data["submit_priority"]),
                submit_action=ActionType(data["submit_action"])
            )
        else:
            return Action(submit_priority="Medium", submit_action="Flag")
    except Exception as e:
        print(f"[DEBUG] LLM error: {e}", flush=True)
        return Action(submit_priority="Medium", submit_action="Flag")

async def run_episode(env, client, task_id):
    history = []
    rewards = []
    obs = await env.reset(task_id)
    step = 1
    done = False
    while not done and step <= MAX_STEPS:
        action = get_model_action(client, task_id, obs.email_text, obs.sender, obs.subject, step, history)
        action_str = action.json()
        obs, reward, done, info = await env.step(action)
        reward_val = reward.value
        rewards.append(reward_val)
        log_step(step, action_str, reward_val, done, info.get("error"))
        history.append(f"Step {step}: {action_str} -> {reward_val:.2f}")
        step += 1
    state = await env.state()
    final_score = state.final_score if state.final_score is not None else 0.0
    success = final_score >= 0.5
    return {"success": success, "steps": step-1, "score": final_score, "rewards": rewards}

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EmailTriageEnv()
    for task_id in TASK_NAMES:
        log_start(task=task_id, env="email_triage_env", model=MODEL_NAME)
        res = await run_episode(env, client, task_id)
        log_end(success=res["success"], steps=res["steps"], score=res["score"], rewards=res["rewards"])
        print()

if __name__ == "__main__":
    asyncio.run(main())