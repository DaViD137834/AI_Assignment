import os
import random
import numpy as np
import pandas as pd
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

# Load reviews dynamically from CSV dataset
@st.cache_data
def load_dataset_reviews():
    csv_path = os.path.join(BASE_DIR, 'cleaned_dataset.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Use original 'review' column if present, otherwise 'clean_text'
        col = 'review' if 'review' in df.columns else 'clean_text'
        return df[col].dropna().tolist()
    else:
        # Fallback pool if cleaned_dataset.csv is not uploaded to GitHub
        return [
            "This movie was an absolute masterpiece! Incredible acting and brilliant plot.",
            "Complete waste of time. Boring storyline and terrible execution.",
            "Visually stunning, though the pacing in the second act was a bit slow.",
            "Hands down one of the worst movies I have ever watched in my life.",
            "10 out of 10! Highly recommend watching this with family and friends.",
            "The soundtrack was amazing, but the ending made zero sense.",
            "Dreadful dialogue and horrible performances across the board."
        ]

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
    .main-title { font-size: 2.2rem; font-weight: 700; color: #38BDF8; text-align: center; margin-bottom: 0.2rem; }
    .sub-title { font-size: 1rem; color: #94A3B8; text-align: center; margin-bottom: 2rem; }
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

# Initialize Non-Repeating Review Deck in Session State
all_reviews = load_dataset_reviews()

if "review_deck" not in st.session_state or len(st.session_state["review_deck"]) == 0:
    deck = all_reviews.copy()
    random.shuffle(deck)
    st.session_state["review_deck"] = deck

if "review_input" not in st.session_state:
    st.session_state["review_input"] = ""

# Callback: Pop one non-repeating review off the deck on each click
def pick_next_unique_review():
    if len(st.session_state["review_deck"]) == 0:
        deck = all_reviews.copy()
        random.shuffle(deck)
        st.session_state["review_deck"] = deck
    
    st.session_state["review_input"] = st.session_state["review_deck"].pop()

st.button("🎲 Pick Random Review from Dataset", on_click=pick_next_unique_review, use_container_width=True)

user_text = st.text_area("Movie Review Input:", key="review_input", placeholder="Type or paste any sentence or review here...", height=130)

if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("⚠️ Please enter text before analyzing.")
    else:
        with st.spinner("Analyzing text..."):
            cleaned_input = clean_text(user_text)
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
                else: # SVM
                    pred = svm_model.predict(vec)[0]
                    distance = svm_model.decision_function(vec)[0]
                    confidence_score = (1 / (1 + np.exp(-abs(distance)))) * 100
                    confidence_text = f"Confidence Score: {confidence_score:.1f}%"

                sentiment = "Positive" if pred == 1 else "Negative"

            elif model_option == "Transformer (DistilBERT)":
                pipe = load_transformer()
                out = pipe(user_text)[0]
                sentiment = out['label'].capitalize()
                confidence_text = f"Confidence Score: {out['score']*100:.1f}%"

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
