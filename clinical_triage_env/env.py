import copy
from typing import Dict, List, Optional, Tuple
from .models import Observation, Action, Reward, State, TriageLevel, TestType, QuestionType
from .tasks import get_random_case_by_difficulty, grade_submission
from .tasks import PatientCase

class ClinicalTriageEnv:
    def __init__(self):
        self.current_case: Optional[PatientCase] = None
        self.step_count: int = 0
        self.done: bool = False
        self.tests_ordered: List[TestType] = []
        self.questions_asked: List[QuestionType] = []
        self.final_triage: Optional[TriageLevel] = None
        self.final_score: Optional[float] = None
        self.max_steps: int = 10
        self.deterioration: float = 0.0

    async def reset(self, task_id: str = "easy") -> Observation:
        # Load a fresh case based on task difficulty (randomised each episode)
        self.current_case = get_random_case_by_difficulty(task_id)
        self.step_count = 0
        self.done = False
        self.tests_ordered = []
        self.questions_asked = []
        self.final_triage = None
        self.final_score = None
        self.deterioration = 0.0
        # Reset revealed symptoms
        self.current_case.revealed_symptoms = []
        return self._get_observation()

    async def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        if self.done:
            raise RuntimeError("Episode already finished")
        reward_value = 0.0
        info = {"error": None}

        # Apply deterioration before processing action (simulates time passing)
        self.current_case.apply_deterioration(self.step_count + 1)
        self.deterioration = min(1.0, self.deterioration + 0.05)

        if action.order_test is not None:
            test = action.order_test
            self.tests_ordered.append(test)
            # Test cost (time penalty)
            reward_value -= 0.05
            if test in self.current_case.relevant_tests:
                reward_value += 0.2
            else:
                reward_value -= 0.1
            self.step_count += 1

        elif action.ask_question is not None:
            q = action.ask_question
            self.questions_asked.append(q)
            # Reveal hidden symptom if question is relevant
            if q in self.current_case.hidden_symptoms:
                symptom = self.current_case.hidden_symptoms[q]
                if symptom not in self.current_case.revealed_symptoms:
                    self.current_case.revealed_symptoms.append(symptom)
                    reward_value += 0.15
            else:
                reward_value -= 0.05
            self.step_count += 1

        elif action.assign_triage is not None:
            self.final_triage = action.assign_triage
            self.final_score = grade_submission(
                self.current_case,
                self.final_triage,
                self.tests_ordered,
                self.questions_asked,
                self.step_count + 1,
                self.deterioration
            )
            reward_value = self.final_score
            self.done = True
            self.step_count += 1

        else:
            reward_value -= 0.2
            info["error"] = "Invalid action"
            self.step_count += 1

        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            self.final_score = 0.01
            reward_value = 0.01
            info["error"] = "Max steps reached"

        return self._get_observation(), Reward(value=reward_value), self.done, info

    async def state(self) -> State:
        return State(
            patient_id=self.current_case.case_id if self.current_case else "",
            step_count=self.step_count,
            done=self.done,
            tests_ordered=self.tests_ordered,
            questions_asked=self.questions_asked,
            final_triage=self.final_triage,
            final_score=self.final_score,
            deterioration=self.deterioration,
            revealed_symptoms=self.current_case.revealed_symptoms if self.current_case else []
        )

    def _get_observation(self) -> Observation:
        if self.current_case is None:
            raise RuntimeError("Environment not reset")
        return Observation(
            age=self.current_case.age,
            heart_rate=self.current_case.heart_rate,
            systolic_bp=self.current_case.systolic_bp,
            oxygen_saturation=self.current_case.oxygen_saturation,
            temperature=self.current_case.temperature,
            pain_level=self.current_case.pain_level,
            chronic_disease_count=self.current_case.chronic_disease_count,
            previous_er_visits=self.current_case.previous_er_visits,
            arrival_mode=self.current_case.arrival_mode,
            chief_complaint=self.current_case.chief_complaint,
            medical_history=self.current_case.medical_history,
            step_count=self.step_count,
            max_steps=self.max_steps,
            deterioration=self.deterioration,
            revealed_symptoms=self.current_case.revealed_symptoms
        )