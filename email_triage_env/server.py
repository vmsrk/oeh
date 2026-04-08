import uvicorn
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel

from email_triage_env import EmailTriageEnv
from email_triage_env.models import Action

app = FastAPI(title="Email Triage Environment")
env = EmailTriageEnv()

class StepRequest(BaseModel):
    action: Action

@app.post("/reset")
async def reset_endpoint(request: Request):
    """Accepts empty body (null) or JSON body with optional task_id."""
    try:
        body = await request.json()
    except json.JSONDecodeError:
        body = {}
    task_id = body.get("task_id", "easy") if isinstance(body, dict) else "easy"
    obs = await env.reset(task_id)
    return {"observation": obs.dict()}

@app.post("/step")
async def step_endpoint(req: StepRequest):
    obs, reward, done, info = await env.step(req.action)
    return {
        "observation": obs.dict(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }

@app.get("/state")
async def state_endpoint():
    state = await env.state()
    return state.dict()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {"message": "Email Triage Environment running"}

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()