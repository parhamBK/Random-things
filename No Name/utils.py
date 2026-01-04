from sklearn.feature_extraction.text import TfidfVectorizer
from symptom_db import DISEASE_SYMPTOMS

def normalize_text(text):
    return text.strip().lower().replace("  ", " ")

def build_table():
    table = {}
    for value in DISEASE_SYMPTOMS.values():
        symptoms = value.split()
        for symptom in symptoms:
            first_word = symptom.split("-")[0]
            table[first_word] = symptom

    return table

def reconstruct_symptoms(word_list):
    result = []
    check_table = build_table()
    for i in word_list:
        if i in check_table.keys():
            result.append(check_table[i])
    return result

def split_symptoms(input_text):

    Symptoms_list = [] #find symptoms
    for Symptoms in DISEASE_SYMPTOMS.values():
        Symptoms_list.append(Symptoms)
    vectorizer = TfidfVectorizer()
    vectorizer.fit_transform(Symptoms_list)
    key_symptoms = vectorizer.get_feature_names_out()

    text = normalize_text(input_text)

    matched_symptom = []
    clean_text = normalize_text(input_text).replace("-", " ").replace("_", " ")

    for symptom in key_symptoms:
        variants = [symptom, symptom.replace("-", " ")] 
        for variant in variants:
            if variant in clean_text:
                matched_symptom.append(symptom)
                break  # avoid duplicate

    print(f"11 {matched_symptom}")
    final_matched_symptom = reconstruct_symptoms(matched_symptom)

    return final_matched_symptom
