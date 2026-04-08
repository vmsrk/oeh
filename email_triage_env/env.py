from typing import Dict, List, Optional, Tuple
from .models import Observation, Action, Reward, State, InfoType, Priority, ActionType
from .tasks import TASKS, Task, grade_submission

class EmailTriageEnv:
    def __init__(self):
        self.current_task: Optional[Task] = None
        self.step_count: int = 0
        self.done: bool = False
        self.requested_info: List[InfoType] = []
        self.max_steps: int = 10
        self.final_score: Optional[float] = None

    async def reset(self, task_id: str = "easy") -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id: {task_id}")
        self.current_task = TASKS[task_id]
        self.step_count = 0
        self.done = False
        self.requested_info = []
        self.final_score = None
        return self._get_observation()

    async def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already finished")
        reward_value = 0.0
        info = {"error": None}

        if action.request_info is not None:
            info_type = action.request_info
            if info_type in self.requested_info:
                reward_value -= 0.1
                info["error"] = f"Already requested {info_type}"
            else:
                self.requested_info.append(info_type)
                if info_type in self.current_task.useful_info:
                    reward_value += 0.2
            self.step_count += 1

        elif action.submit_priority is not None and action.submit_action is not None:
            self.final_score = grade_submission(
                self.current_task,
                action.submit_priority,
                action.submit_action,
                self.step_count + 1,
                self.requested_info,
            )
            reward_value = self.final_score
            self.done = True
            self.step_count += 1
        else:
            reward_value -= 0.2
            info["error"] = "Invalid action: must provide request_info or both submit_priority and submit_action"
            self.step_count += 1

        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            self.final_score = 0.0
            reward_value = 0.0
            info["error"] = "Max steps reached"

        return self._get_observation(), Reward(value=reward_value), self.done, info

    async def state(self) -> State:
        return State(
            task_id=self.current_task.task_id if self.current_task else "",
            step_count=self.step_count,
            done=self.done,
            info_already_requested=self.requested_info,
            final_score=self.final_score,
        )

    def _get_observation(self) -> Observation:
        if self.current_task is None:
            raise RuntimeError("Environment not reset")
        return Observation(
            email_text=self.current_task.email_text,
            sender=self.current_task.sender,
            subject=self.current_task.subject,
            available_info_types=self.current_task.useful_info,
            step_count=self.step_count,
            max_steps=self.max_steps,
        )