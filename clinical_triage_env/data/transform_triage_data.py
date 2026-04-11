import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv("synthetic_medical_triage.csv")

# Rename columns to match expected internal names
df.rename(columns={
    'systolic_blood_pressure': 'sbp',
    'oxygen_saturation': 'spo2',
    'body_temperature': 'temp',
    'pain_level': 'pain',
    'chronic_disease_count': 'chronic_count',
    'previous_er_visits': 'prev_visits'
}, inplace=True)

# Add case_id
df['case_id'] = [f"case_{i:03d}" for i in range(len(df))]

# Add difficulty based on triage_level: 0=hard, 1=medium, 2=medium, 3=easy
def get_difficulty(triage):
    if triage == 0:
        return "hard"
    elif triage == 1:
        return "medium"
    elif triage == 2:
        return "medium"
    else:
        return "easy"
df['difficulty'] = df['triage_level'].apply(get_difficulty)

# Generate synthetic chief complaint
def generate_chief_complaint(row):
    triage = row['triage_level']
    pain = row['pain']
    temp = row['temp']
    if triage == 0:
        return "Sudden severe chest pain radiating to arm, diaphoretic"
    elif triage == 1:
        if row['spo2'] < 92:
            return f"Shortness of breath, oxygen saturation {row['spo2']}%"
        else:
            return "Severe abdominal pain and vomiting"
    elif triage == 2:
        if temp > 38.0:
            return f"Fever of {temp}°C, productive cough for 3 days"
        else:
            return f"Moderate headache and dizziness, pain level {pain}"
    else:
        options = ["sore throat", "ankle sprain", "mild cough"]
        return f"Mild {np.random.choice(options)} for 2 days"
df['chief_complaint'] = df.apply(generate_chief_complaint, axis=1)

# Generate synthetic medical history
def generate_medical_history(row):
    chronic = row['chronic_count']
    if chronic == 0:
        return "No significant past medical history"
    elif chronic == 1:
        return "Hypertension"
    elif chronic == 2:
        return "Type 2 diabetes, hypertension"
    else:
        return "COPD, coronary artery disease, diabetes"
df['medical_history'] = df.apply(generate_medical_history, axis=1)

# Select and order the final columns (as required by tasks.py)
final_columns = [
    'case_id', 'difficulty', 'age', 'heart_rate', 'sbp', 'spo2', 'temp',
    'pain', 'chronic_count', 'prev_visits', 'arrival_mode',
    'chief_complaint', 'medical_history', 'triage_level'
]
df_final = df[final_columns]

# Save to CSV
df_final.to_csv("patient_cases.csv", index=False)
print(f"Successfully saved {len(df_final)} cases to patient_cases.csv")