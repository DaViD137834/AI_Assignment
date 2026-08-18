import os
import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned)

@st.cache_resource
def load_ml_resources():
    vec_path = os.path.join(BASE_DIR, 'vectorizer.pkl')
    nb_path = os.path.join(BASE_DIR, 'nb_model.pkl')
    svm_path = os.path.join(BASE_DIR, 'svm_model.pkl')
    
    vectorizer = joblib.load(vec_path)
    nb_model = joblib.load(nb_path)
    svm_model = joblib.load(svm_path)
    return vectorizer, nb_model, svm_model

@st.cache_resource
def load_transformer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

st.set_page_config(page_title="AI Sentiment Studio", page_icon="🎬", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; text-align: center; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1rem; color: #64748B; text-align: center; margin-bottom: 2rem; }
    .result-card-pos { background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 12px; padding: 20px; text-align: center; margin-top: 15px; }
    .result-card-neg { background-color: #FEF2F2; border: 2px solid #EF4444; border-radius: 12px; padding: 20px; text-align: center; margin-top: 15px; }
    .result-title-pos { color: #15803D; font-size: 1.5rem; font-weight: 700; }
    .result-title-neg { color: #B91C1C; font-size: 1.5rem; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Test Natural Language Processing algorithms in real-time</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Model Configuration")
model_option = st.sidebar.selectbox(
    "Select Model Architecture:",
    ["Naïve Bayes", "Support Vector Machine (SVM)", "Transformer (DistilBERT)"]
)

user_text = st.text_area("Movie Review Input:", placeholder="Type or paste any sentence or review here...", height=130)

st.write("💡 **Try clicking a sample sentence:**")
col_s1, col_s2, col_s3 = st.columns(3)
if col_s1.button("Positive Sentence"):
    user_text = "This movie had an unbelievable plot and magnificent visual effects."
if col_s2.button("Negative Sentence"):
    user_text = "The acting was dreadful and I wanted to leave the theater early."
if col_s3.button("Conversational Sentence"):
    user_text = "Hello, how are you feeling today?"

if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("⚠️ Please enter text before analyzing.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned_input = clean_text(user_text)
            
            # Fallback to original text if stop-word removal leaves text empty
            input_for_ml = cleaned_input if cleaned_input.strip() else user_text.lower()

            sentiment = ""
            confidence_text = ""

            if model_option in ["Naïve Bayes", "Support Vector Machine (SVM)"]:
                vectorizer, nb_model, svm_model = load_ml_resources()
                vec = vectorizer.transform([input_for_ml])

                if model_option == "Naïve Bayes":
                    pred = nb_model.predict(vec)[0]
                    proba = nb_model.predict_proba(vec)[0]
                    confidence_text = f"Confidence Score: {max(proba)*100:.1f}%"
                else:
                    pred = svm_model.predict(vec)[0]
                    confidence_text = "Classification Method: Linear SVM Boundary"

                sentiment = "Positive" if pred == 1 else "Negative"

            elif model_option == "Transformer (DistilBERT)":
                pipe = load_transformer()
                out = pipe(user_text)[0]
                sentiment = out['label'].capitalize()
                confidence_text = f"Confidence Score: {out['score']*100:.1f}%"

        # Display Result
        if sentiment == "Positive":
            st.markdown(f"""
                <div class="result-card-pos">
                    <div class="result-title-pos">😊 POSITIVE SENTIMENT</div>
                    <p style="color: #166534; margin-top: 5px; font-weight: 500;">{confidence_text}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card-neg">
                    <div class="result-title-neg">🙁 NEGATIVE SENTIMENT</div>
                    <p style="color: #991B1B; margin-top: 5px; font-weight: 500;">{confidence_text}</p>
                </div>
            """, unsafe_allow_html=True)
