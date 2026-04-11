from typing import Literal, Optional, List
from pydantic import BaseModel

TriageLevel = Literal[0, 1, 2, 3]
TestType = Literal["CBC", "Troponin", "ChestXRay", "CTHead", "Lactate", "Urinalysis"]
QuestionType = Literal["symptom_duration", "pain_location", "medication_allergy", "past_medical_history"]

class Observation(BaseModel):
    age: int
    heart_rate: int
    systolic_bp: int
    oxygen_saturation: int
    temperature: float
    pain_level: int
    chronic_disease_count: int
    previous_er_visits: int
    arrival_mode: str
    chief_complaint: str
    medical_history: str
    step_count: int
    max_steps: int = 10
    deterioration: float          # 0.0 = stable, 1.0 = critical
    revealed_symptoms: List[str]  # symptoms uncovered by asking questions

class Action(BaseModel):
    order_test: Optional[TestType] = None
    ask_question: Optional[QuestionType] = None
    assign_triage: Optional[TriageLevel] = None

class Reward(BaseModel):
    value: float

class State(BaseModel):
    patient_id: str
    step_count: int
    done: bool
    tests_ordered: List[TestType]
    questions_asked: List[QuestionType]
    final_triage: Optional[TriageLevel] = None
    final_score: Optional[float] = None
    deterioration: float
    revealed_symptoms: List[str]