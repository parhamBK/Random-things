from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def suggest_conditions(user_symptoms, disease_db, top_n=3):
    all_text = list(disease_db.values()) + [" ".join(user_symptoms)]
    labels = list(disease_db.keys())

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)

    user_vector = tfidf_matrix[-1]
    disease_vectors = tfidf_matrix[:-1]

    similarities = cosine_similarity(user_vector, disease_vectors).flatten()
    ranked = sorted(zip(labels, similarities), key=lambda x: x[1], reverse=True)

    return [(disease, round(score, 2)) for disease, score in ranked[:top_n] if score > 0.0]
