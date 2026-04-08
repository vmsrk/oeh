from typing import List, Dict
from .models import Priority, ActionType, InfoType   # ← added dot

class Task:
    def __init__(self, task_id: str, email_text: str, sender: str, subject: str,
                 expected_priority: Priority, expected_action: ActionType,
                 useful_info: List[InfoType], difficulty: str):
        self.task_id = task_id
        self.email_text = email_text
        self.sender = sender
        self.subject = subject
        self.expected_priority = expected_priority
        self.expected_action = expected_action
        self.useful_info = useful_info
        self.difficulty = difficulty

TASKS: Dict[str, Task] = {
    "easy": Task(
        task_id="easy",
        email_text="""Dear Support,
I no longer wish to receive your newsletter. Please remove me from your mailing list immediately.
Thank you.""",
        sender="newsletter_unsubscribe@example.com",
        subject="Unsubscribe",
        expected_priority="Low",
        expected_action="Archive",
        useful_info=[],
        difficulty="easy",
    ),
    "medium": Task(
        task_id="medium",
        email_text="""Subject: Defective product - Order #12345

I received my wireless mouse yesterday, but it won't turn on. I've tried new batteries and holding the power button. This is unacceptable for a $50 product. Please send a replacement or refund me immediately.

Regards,
John Doe""",
        sender="john.doe@example.com",
        subject="Defective product - Order #12345",
        expected_priority="High",
        expected_action="Reply",
        useful_info=["product_details", "previous_tickets"],
        difficulty="medium",
    ),
    "hard": Task(
        task_id="hard",
        email_text="""Dear Customer,
Your PayPal account has been limited due to suspicious activity. Please click here to verify your identity within 24 hours or your account will be suspended.

https://paypal-verify-security.com/login

Sincerely,
PayPal Security Team""",
        sender="security@paypal.com",
        subject="Urgent: Account Limited",
        expected_priority="Medium",
        expected_action="Flag",
        useful_info=["sender_history", "previous_tickets"],
        difficulty="hard",
    ),
}


def grade_submission(task: Task, chosen_priority: Priority, chosen_action: ActionType,
                     steps_taken: int, requested_info: List[InfoType]) -> float:
    score = 0.0
    if chosen_priority == task.expected_priority:
        score += 0.4
    if chosen_action == task.expected_action:
        score += 0.4
    if steps_taken <= 3:
        score += 0.2
    irrelevant = [info for info in requested_info if info not in task.useful_info]
    penalty = min(0.2, len(irrelevant) * 0.05)
    score = max(0.0, min(1.0, score - penalty))

    # Clamp to strictly between 0 and 1
    if score <= 0.0:
        score = 0.01
    if score >= 1.0:
        score = 0.99
    return score