---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Email Triage Environment – OpenEnv

A real-world environment where an AI agent must triage customer support emails. The agent can request additional information (sender history, product details, previous tickets) and then submit a priority (High/Medium/Low) and an action (Reply/Archive/Delete/Flag). Partial rewards are given for useful info requests; final reward is based on decision accuracy and efficiency.

## Motivation
Email triage is a common human task in customer support. This environment helps train/evaluate agents on:
- Information gathering (asking for relevant context)
- Decision making under uncertainty
- Balancing speed vs. accuracy

## Action & Observation Spaces

### Observation
- `email_text` (str): Body of the email
- `sender` (str): Sender email address
- `subject` (str): Email subject line
- `available_info_types` (list): Which info types are useful for this task
- `step_count` (int): Current step number
- `max_steps` (int): Episode limit (default 10)

### Action (one of two forms)
1. **Request information**: `{"request_info": "sender_history"}`  
   Options: `sender_history`, `product_details`, `previous_tickets`
2. **Submit decision**: `{"submit_priority": "High", "submit_action": "Reply"}`  
   Priority: `High`, `Medium`, `Low`  
   Action: `Reply`, `Archive`, `Delete`, `Flag`

## Tasks & Graders

| Task   | Difficulty | Description | Expected Priority | Expected Action |
|--------|------------|-------------|-------------------|----------------|
| easy   | Easy       | Unsubscribe request | Low | Archive |
| medium | Medium     | Defective product complaint | High | Reply |
| hard   | Hard       | Phishing-like PayPal alert | Medium | Flag |

**Grader formula** (score 0.0–1.0):
- 0.4 for correct priority
- 0.4 for correct action
- 0.2 if submitted within ≤3 steps
- Penalty up to -0.2 for irrelevant info requests

## Reward Function
- `+0.2` for each **useful** info request (defined per task)
- `-0.1` for repeating the same request
- Final reward = overall grade (0.0–1.0) when submitting
- Episode ends on `submit` or after max steps (then score 0)

## Setup & Usage

### Local installation
```bash
git clone <your-repo-url>
cd email_triage_env
pip install -r requirements.txt