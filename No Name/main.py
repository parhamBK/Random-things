from symptom_db import DISEASE_SYMPTOMS
from smart_engine import suggest_conditions
from utils import normalize_text, split_symptoms

print("Hello! I'm your Smart Hello Medical Bot.")
print("enter your symptoms, writ it with your own (e.g., fever, cough, headache):")

user_input = input("> ")
symptoms = split_symptoms(user_input)

# Save to log
with open("log.txt", "a", encoding="utf-8") as f:
    f.write("User symptoms: " + ", ".join(symptoms) + "\n")

results = suggest_conditions(symptoms, DISEASE_SYMPTOMS)

# Display results
if results:
    print("\n Based on your symptoms, you may have:")
    for disease, score in results:
        print(f" {disease.capitalize()} (Similarity: {int(score * 100)}%)")
else:
    print(" Sorry, I couldn't find any matching conditions in my knowledge base.")
