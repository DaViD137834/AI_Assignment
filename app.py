import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# Download necessary NLTK corpora
nltk.download('stopwords')
nltk.download('wordnet')

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
def load_transformer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# UI Configuration
st.set_page_config(page_title="Movie Sentiment Analyzer", page_icon="🎬")
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Select a group member's algorithm to classify any movie review in real-time!")

# Model Selection Menu
model_option = st.sidebar.selectbox(
    "Select Algorithm / Group Member",
    ["Member 1: Naïve Bayes", "Member 2: Support Vector Machine (SVM)", "Member 3: Transformer (DistilBERT)"]
)

user_text = st.text_area("Enter Movie Review:", placeholder="Type your movie review here...")

if st.button("Predict Sentiment"):
    if not user_text.strip():
        st.warning("Please enter a review first!")
    else:
        cleaned_input = clean_text(user_text)

        if "Member 1" in model_option:
            vectorizer = joblib.load('vectorizer.pkl')
            model = joblib.load('nb_model.pkl')
            vec = vectorizer.transform([cleaned_input])
            pred = model.predict(vec)[0]
            result = "Positive Review 😊" if pred == 1 else "Negative Review 🙁"

        elif "Member 2" in model_option:
            vectorizer = joblib.load('vectorizer.pkl')
            model = joblib.load('svm_model.pkl')
            vec = vectorizer.transform([cleaned_input])
            pred = model.predict(vec)[0]
            result = "Positive Review 😊" if pred == 1 else "Negative Review 🙁"

        elif "Member 3" in model_option:
            pipe = load_transformer()
            out = pipe(user_text)[0]
            label = out['label'].capitalize()
            result = f"{label} Review " + ("😊" if label == "Positive" else "🙁")

        st.success(f"**Result ({model_option}):** {result}")
