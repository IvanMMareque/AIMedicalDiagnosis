from experta import *
import pandas as pd

# Load preprocessed dataset
csv_path = "preprocessed_disease_dataset.csv"
df = pd.read_csv(csv_path)

class MedicalExpert(KnowledgeEngine):
    def __init__(self):
        super().__init__()
        self.diagnosis = None

    @DefFacts()
    def initial_facts(self):
        """Base fact to trigger diagnosis"""
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
                print(f"Expert System: The patient may have {disease}.")
            
            setattr(MedicalExpert, f'rule_{disease_name.replace(" ", "_")}', diagnose_disease)

# Generate rules before instantiating the engine
generate_rules()

def run_expert_system(symptom_inputs):
    engine = MedicalExpert()
    engine.reset()

    # Normalize user symptoms (convert to lowercase + strip spaces)
    symptom_inputs = [symptom.lower().strip() for symptom in symptom_inputs]

    # Debugging: Print user symptoms
    print("\n=== Debug: Checking User Symptoms ===")
    print(f"User input symptoms (normalized): {symptom_inputs}")

    # Print disease symptoms from dataset
    print("\n=== Debug: Checking Disease Symptoms in Dataset ===")
    for _, row in df.iterrows():
        disease_name = row["Disease"]
        disease_symptoms = [col.lower().strip() for col in df.columns if row[col] == 1]
        print(f"{disease_name}: {disease_symptoms}")
    print("\n=== End of Debug ===\n")

    # Declare symptoms inputted by the user
    for symptom in symptom_inputs:
        engine.declare(Fact(symptom=symptom))

    engine.run()

    # Extract exact matches from the inference engine
    diagnosed_conditions = [fact['diagnosis'] for fact in engine.facts.values() if 'diagnosis' in fact]
    
    # If no exact match, try fuzzy matching (at least 60% match)
    if not diagnosed_conditions:
        possible_diagnoses = []
        for _, row in df.iterrows():
            disease_name = row["Disease"]
            disease_symptoms = [col.lower().strip() for col in df.columns if row[col] == 1]

            # Allow fuzzy matching: Symptoms are matched even if not exact
            match_count = sum(1 for symptom in symptom_inputs if any(symptom in disease_symptom for disease_symptom in disease_symptoms))
            match_percentage = (match_count / len(symptom_inputs)) * 100 if symptom_inputs else 0

            # If at least 60% of symptoms match, consider it a possible diagnosis
            if match_percentage >= 60:
                possible_diagnoses.append((disease_name, match_percentage))

        # Sort diagnoses by best match percentage
        possible_diagnoses.sort(key=lambda x: x[1], reverse=True)

        if possible_diagnoses:
            print("\nPossible Diagnoses (Ranked by Symptom Match):")
            for disease, confidence in possible_diagnoses:
                print(f"- {disease} ({confidence:.1f}% match)")
            return [d[0] for d in possible_diagnoses]  # Return disease names only

    # If still no diagnosis found
    print("No clear diagnosis. More data needed.")
    return diagnosed_conditions if diagnosed_conditions else None

# Example Usage
if __name__ == "__main__":
    user_symptoms = ["Fever", "Cough", "Sore throat"]
    print("User Symptoms:", user_symptoms)
    diagnosis = run_expert_system(user_symptoms)

    if diagnosis:
        print(f"Final Diagnosis: {', '.join(diagnosis)}")
    else:
        print("No clear diagnosis. More data needed.")