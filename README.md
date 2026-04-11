---
title: Clinical Triage Nurse
emoji: 🏥
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# Clinical Triage Nurse – OpenEnv

Emergency department triage simulation. Agent assesses vital signs, chief complaint, and medical history, then orders tests, asks questions, and assigns a triage level (0=critical to 3=low priority).

## Real-world utility
Used to train/evaluate AI for emergency triage – a genuine, high‑stakes medical task.

## Tasks
| Task   | Difficulty | Description |
|--------|------------|-------------|
| easy   | Easy       | Sore throat, stable → triage 3 |
| medium | Medium     | COPD exacerbation, hypoxia → triage 1 |
| hard   | Hard       | Acute MI, hypotension → triage 0 |

## API
- `POST /reset` – start episode
- `POST /step`  – send action (order_test, ask_question, assign_triage)
- `GET /state`  – current state

## Local run
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python inference.py