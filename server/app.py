import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Optional

from clinical_triage_env import ClinicalTriageEnv
from clinical_triage_env.models import Action

app = FastAPI()
env = ClinicalTriageEnv()

class ResetRequest(BaseModel):
    task_id: str = "easy"   # default value

class StepRequest(BaseModel):
    action: Action

@app.post("/reset")
async def reset_endpoint(req: Optional[ResetRequest] = None):
    # If no body or empty body, use default
    if req is None:
        req = ResetRequest()
    obs = await env.reset(req.task_id)
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

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()