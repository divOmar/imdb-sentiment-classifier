import streamlit as st
st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬")
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------
# Load saved model + vectorizer
# ---------------------------
@st.cache_resource
def load_sentiment_model():
    model = load_model("imdb_sentiment_analysis_model.h5")
    vectorizer = joblib.load("tfidf.pkl")
    return model, vectorizer

model, tfidf = load_sentiment_model()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Type or paste a movie review, and the model will predict whether it's **positive** or **negative**.")

# Input box
review = st.text_area("✍️ Enter your review here:", height=150)

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Preprocess + vectorize
        X_input = tfidf.transform([review])
        prediction = model.predict(X_input)[0][0]

        if prediction > 0.5:
            st.success(f"✅ Positive review (confidence: {prediction:.2f})")
        else:
            st.error(f"❌ Negative review (confidence: {1 - prediction:.2f})")






