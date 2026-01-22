import streamlit as st
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# CONFIGURARE
FILE_DATASET = 'dataset.txt'
FILE_VECTORIZER = 'bot_vectorizer.pkl'
FILE_MODEL = 'bot_svd_model.pkl'
FILE_DATA = 'bot_data.pkl'

st.set_page_config(page_title="AI Chatbot All-in-One", page_icon="🤖")


# 1. FUNCȚIA DE ANTRENARE
def antreneaza_model():
    """Citește dataset.txt și regenerează fișierele .pkl"""
    if not os.path.exists(FILE_DATASET):
        return False, "Fișierul dataset.txt lipsește!"

    # Citire date
    questions = []
    answers = []

    try:
        with open(FILE_DATASET, 'r', encoding='utf-8') as f:
            for linie in f:
                if '|' in linie and linie.strip():
                    parts = linie.split('|')
                    questions.append(parts[0].strip())
                    answers.append(parts[1].strip())
    except Exception as e:
        return False, f"Eroare la citire: {e}"

    if len(questions) == 0:
        return False, "Dataset-ul este gol!"

    # Procesare
    # Folosim setările avansate (char_wb) pentru a recunoaște variații de cuvinte
    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='char_wb', ngram_range=(3, 5))
    X = vectorizer.fit_transform(questions)

    # Logică SVD vs TF-IDF simplu
    use_svd = False
    matrix_final = X
    lsa = None

    if len(questions) > 10:
        use_svd = True
        n_components = min(100, len(questions) - 1)
        lsa = TruncatedSVD(n_components=n_components)
        matrix_final = lsa.fit_transform(X)

    # Salvare pe disc
    joblib.dump(vectorizer, FILE_VECTORIZER)

    data_to_save = {'matrix': matrix_final, 'answers': answers, 'use_svd': use_svd}
    joblib.dump(data_to_save, FILE_DATA)

    if use_svd:
        joblib.dump(lsa, FILE_MODEL)

    return True, f"Antrenare reușită pe {len(questions)} exemple!"


# 2. FUNCȚIA DE ÎNCĂRCARE (Cached)
@st.cache_resource
def incarca_resurse():
    """Încarcă modelele în memorie. Dacă nu există, le antrenează întâi."""

    # Verificăm dacă fișierele există. Dacă nu, antrenăm acum.
    if not os.path.exists(FILE_VECTORIZER) or not os.path.exists(FILE_DATA):
        success, msg = antreneaza_model()
        if not success:
            return None, None, None, None, None  # Eroare critică

    # Încărcăm resursele
    vectorizer = joblib.load(FILE_VECTORIZER)
    data = joblib.load(FILE_DATA)
    matrix_final = data['matrix']
    answers = data['answers']
    use_svd = data.get('use_svd', False)

    lsa = None
    if use_svd:
        lsa = joblib.load(FILE_MODEL)

    return vectorizer, lsa, matrix_final, answers, use_svd

# 3. LOGICA DE RĂSPUNS
def get_best_response(user_input, vectorizer, lsa, matrix_final, answers, use_svd):
    try:
        user_vec = vectorizer.transform([user_input])

        if use_svd:
            query_vec = lsa.transform(user_vec)
        else:
            query_vec = user_vec

        similarities = cosine_similarity(query_vec, matrix_final)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > 0.7:
            return answers[best_idx]
        else:
            return "Nu sunt sigur că am înțeles. Poți reformula?"
    except Exception:
        return "Eroare la procesarea răspunsului."


# 4. INTERFAȚA GRAFICĂ (Streamlit)

# Sidebar pentru control (Re-antrenare)
with st.sidebar:
    st.header("⚙️ Panou Control")
    st.info("Modifică 'dataset.txt' și apasă butonul de mai jos.")

    if st.button("🔄 Re-antrenează Modelul"):
        with st.spinner("Se învață noile date..."):
            # Ștergem cache-ul vechi
            st.cache_resource.clear()
            # Rulăm antrenarea
            ok, mesaj = antreneaza_model()
            if ok:
                st.success(mesaj)
            else:
                st.error(mesaj)
            # Reîncărcăm pagina
            st.rerun()

st.title("🤖 Asistent Inteligent")

# Încărcăm "Creierul"
vectorizer, lsa, matrix_final, answers, use_svd = incarca_resurse()

if vectorizer is None:
    st.error("Eroare Critică: Nu pot încărca sau antrena modelul. Verifică `dataset.txt`.")
    st.stop()

# Gestionarea istoricului de chat
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Salut! Cu ce te pot ajuta?"}]

# Afișarea mesajelor
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input utilizator
if prompt := st.chat_input("Scrie mesajul tău..."):
    # 1. Afișăm user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Calculăm răspuns
    raspuns = get_best_response(prompt, vectorizer, lsa, matrix_final, answers, use_svd)

    # 3. Afișăm bot
    st.session_state.messages.append({"role": "assistant", "content": raspuns})
    with st.chat_message("assistant"):
        st.markdown(raspuns)