import os
import csv
import random
from typing import Dict, List
from .models import TriageLevel, TestType, QuestionType


class PatientCase:
    def __init__(self, row: dict):
        self.case_id = row["case_id"]
        self.difficulty = row["difficulty"]
        self.age = int(float(row["age"]))
        self.heart_rate = int(float(row["heart_rate"]))
        self.systolic_bp = int(float(row["sbp"]))
        self.oxygen_saturation = int(float(row["spo2"]))
        self.temperature = float(row["temp"])
        self.pain_level = int(float(row["pain"]))
        self.chronic_disease_count = int(float(row["chronic_count"]))
        self.previous_er_visits = int(float(row["prev_visits"]))
        self.arrival_mode = row["arrival_mode"]
        self.chief_complaint = row["chief_complaint"]
        self.medical_history = row["medical_history"]
        self.correct_triage = int(float(row["triage_level"]))

        # Hidden symptoms that can be revealed by asking questions
        self.hidden_symptoms = self._generate_hidden_symptoms()
        self.revealed_symptoms = []  # will be filled during episode

        # Deterioration rate (how fast vitals worsen per step)
        self.deterioration_rate = 0.02 if self.correct_triage <= 1 else 0.01

        self.relevant_tests = self._derive_relevant_tests()
        self.relevant_questions = self._derive_relevant_questions()

    def _generate_hidden_symptoms(self) -> Dict[QuestionType, str]:
        """Map questions to hidden information."""
        base = {}
        if "chest" in self.chief_complaint.lower():
            base["pain_location"] = "Substernal chest pain radiating to left arm"
            base["symptom_duration"] = "Started 2 hours ago, worsening"
        elif "breath" in self.chief_complaint.lower():
            base["symptom_duration"] = "Progressive over 3 days"
            base["past_medical_history"] = "Known COPD"
        else:
            base["symptom_duration"] = "2 days"
            base["medication_allergy"] = "No known allergies"
        if self.correct_triage <= 1:
            base["past_medical_history"] = self.medical_history
        return base

    def _derive_relevant_tests(self) -> List[TestType]:
        tests = []
        if self.correct_triage <= 1:
            if "chest" in self.chief_complaint.lower() or self.pain_level >= 7:
                tests.extend(["Troponin", "ChestXRay"])
            if self.oxygen_saturation < 92:
                tests.extend(["CBC", "Lactate"])
            if self.age > 60 and self.chronic_disease_count >= 2:
                tests.append("CTHead")
        elif self.correct_triage == 2:
            if self.temperature > 38.0:
                tests.append("CBC")
            if self.pain_level >= 6:
                tests.append("Urinalysis")
        return list(set(tests))

    def _derive_relevant_questions(self) -> List[QuestionType]:
        questions = []
        if self.correct_triage <= 1:
            questions.extend(["pain_location", "symptom_duration", "past_medical_history"])
        if "allergy" not in self.medical_history.lower():
            questions.append("medication_allergy")
        return questions

    def apply_deterioration(self, step: int):
        """Worsen vital signs based on steps taken (simulate patient decline)."""
        factor = step * self.deterioration_rate
        self.heart_rate = min(160, self.heart_rate + int(5 * factor))
        self.systolic_bp = max(60, self.systolic_bp - int(3 * factor))
        self.oxygen_saturation = max(70, self.oxygen_saturation - int(1 * factor))
        self.temperature = min(41.0, self.temperature + 0.1 * factor)
        self.pain_level = min(10, self.pain_level + int(1 * factor))


def load_all_cases() -> Dict[str, PatientCase]:
    csv_path = os.path.join(os.path.dirname(__file__), "data", "patient_cases.csv")
    cases = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = PatientCase(row)
            cases[case.case_id] = case
    return cases


ALL_CASES = load_all_cases()


def get_random_case_by_difficulty(difficulty: str) -> PatientCase:
    """Return a random case of the given difficulty (for dynamic episodes)."""
    matching = [c for c in ALL_CASES.values() if c.difficulty == difficulty]
    return random.choice(matching)


# For backward compatibility with tasks defined in openenv.yaml
TASKS = {
    "easy": get_random_case_by_difficulty("easy"),
    "medium": get_random_case_by_difficulty("medium"),
    "hard": get_random_case_by_difficulty("hard"),
}


def grade_submission(case: PatientCase, assigned_triage: TriageLevel,
                     tests_ordered: List[TestType], questions_asked: List[QuestionType],
                     steps_taken: int, deterioration: float) -> float:
    """Enhanced grader that penalises delay and deterioration."""
    score = 0.0

    # Triage accuracy (0.4)
    if assigned_triage == case.correct_triage:
        score += 0.4
    else:
        penalty = 0.2 if assigned_triage < case.correct_triage else 0.1
        score -= penalty

    # Test appropriateness (0.3)
    relevant = set(case.relevant_tests)
    ordered_set = set(tests_ordered)
    correct_tests = len(ordered_set & relevant)
    unnecessary = len(ordered_set - relevant)
    test_score = correct_tests * 0.1 - unnecessary * 0.05
    score += max(-0.3, min(0.3, test_score))

    # Question relevance (0.1)
    relevant_q = set(case.relevant_questions)
    asked_set = set(questions_asked)
    good_q = len(asked_set & relevant_q)
    bad_q = len(asked_set - relevant_q)
    q_score = good_q * 0.05 - bad_q * 0.02
    score += max(-0.1, min(0.1, q_score))

    # Efficiency (0.1)
    if steps_taken <= 3:
        score += 0.1
    elif steps_taken <= 6:
        score += 0.05

    # Penalty for patient deterioration (max -0.1)
    score -= deterioration * 0.1

    # Clamp to (0,1)
    if score <= 0.0:
        score = 0.01
    if score >= 1.0:
        score = 0.99
    return score