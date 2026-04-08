import uvicorn
import json
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel

from email_triage_env import EmailTriageEnv
from email_triage_env.models import Action

app = FastAPI(title="Email Triage Environment")
env = EmailTriageEnv()

class StepRequest(BaseModel):
    action: Action

@app.api_route("/reset", methods=["GET", "POST"])
async def reset_endpoint(
    request: Request,
    task_id: str = Query("easy", description="Task ID for GET requests")
):
    """
    Reset the environment.
    - GET: use query parameter ?task_id=easy
    - POST: accept JSON body {"task_id": "easy"} or empty body (defaults to "easy")
    """
    if request.method == "GET":
        # Use query parameter
        tid = task_id
    else:
        # POST: try to read JSON body, default to "easy" if empty or invalid
        try:
            body = await request.json()
            tid = body.get("task_id", "easy") if isinstance(body, dict) else "easy"
        except (json.JSONDecodeError, TypeError):
            tid = "easy"
    obs = await env.reset(tid)
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