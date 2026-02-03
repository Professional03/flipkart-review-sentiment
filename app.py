import streamlit as st

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('stopwords')
nltk.download('wordnet')

model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

st.markdown(
    """
    <style>
    .stButton>button {
        width: 100%;
        height: 45px;
        font-size: 18px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛍️ Flipkart Review Sentiment Analyzer")
st.caption("Analyze customer reviews and understand sentiment instantly")

user_name = st.text_input("Your Name")

review_text = st.text_area(
    "Enter Product Review",
    height=150,
    placeholder="Type or paste a Flipkart product review here..."
)

if st.button("Analyze Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review to analyze")
    else:
        cleaned = clean_text(review_text)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        name = user_name if user_name else "User"

        if prediction == "Positive":
            st.success(f"Hello {name}! Thank you for your positive feedback ")
        else:
            st.error(f"Hello {name}! We will definitely work on your negative feedback ")
