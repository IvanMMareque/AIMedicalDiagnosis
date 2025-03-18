from flask import Flask, render_template, request
from experta import *
import pandas as pd

# Load preprocessed dataset
csv_path = "preprocessed_disease_dataset.csv"
df = pd.read_csv(csv_path)

# Initialize Flask app
app = Flask(__name__)

class MedicalExpert(KnowledgeEngine):
    @DefFacts()
    def initial_facts(self):
        yield Fact(action="diagnose")

# Dynamically generate rules based on dataset
def generate_rules():
    for _, row in df.iterrows():
        disease_name = row['Disease']
        symptoms = [col for col in df.columns if row[col] == 1]
        
        if symptoms:
            @Rule(*[Fact(symptom=symptom) for symptom in symptoms])
            def diagnose_disease(self, disease=disease_name):
                self.declare(Fact(diagnosis=disease))
            
            setattr(MedicalExpert, f'rule_{disease_name.replace(" ", "_")}', diagnose_disease)

generate_rules()

def run_expert_system(symptom_inputs):
    engine = MedicalExpert()
    engine.reset()

    # Normalize user symptoms (lowercase + strip spaces)
    symptom_inputs = [symptom.lower().strip() for symptom in symptom_inputs]

    # Declare user symptoms
    for symptom in symptom_inputs:
        engine.declare(Fact(symptom=symptom))

    engine.run()

    # Extract exact matches from the inference engine
    diagnosed_conditions = [fact['diagnosis'] for fact in engine.facts.values() if 'diagnosis' in fact]

    # If no exact match, use fuzzy matching (at least 50% match)
    if not diagnosed_conditions:
        possible_diagnoses = []
        for _, row in df.iterrows():
            disease_name = row["Disease"]
            disease_symptoms = [col.lower().strip() for col in df.columns if row[col] == 1]

            # Count matches
            match_count = sum(1 for symptom in symptom_inputs if symptom in disease_symptoms)
            match_percentage = (match_count / len(disease_symptoms)) * 100 if disease_symptoms else 0

            # If 50% or more symptoms match, suggest it
            if match_percentage >= 50:
                possible_diagnoses.append((disease_name, match_percentage))

        # Sort by best match
        possible_diagnoses.sort(key=lambda x: x[1], reverse=True)

        if possible_diagnoses:
            return [d[0] for d in possible_diagnoses]

    return diagnosed_conditions if diagnosed_conditions else ["No clear diagnosis"]

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis = None
    if request.method == 'POST':
        symptoms = request.form.get("symptoms")
        symptoms_list = [s.strip() for s in symptoms.split(",")]
        diagnosis = run_expert_system(symptoms_list)
    
    return render_template("index.html", diagnosis=diagnosis)

if __name__ == '__main__':
    app.run(debug=True)
