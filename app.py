import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


st.set_page_config(
    page_title="Flipkart Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)


@st.cache_resource
def load_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

load_nltk()

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

NEGATION_WORDS = {"not", "no", "never", "n't"}
stop_words = set(stopwords.words("english")) - NEGATION_WORDS
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()

    processed = []
    negate = False

    for word in words:
        if word in NEGATION_WORDS:
            negate = True
            processed.append(word)
            continue

        lemma = lemmatizer.lemmatize(word)

        if negate:
            processed.append("not_" + lemma)
            negate = False
        else:
            if lemma not in stop_words:
                processed.append(lemma)

    return " ".join(processed)


st.markdown(
    """
    <style>
        .stButton > button {
            width: 100%;
            height: 45px;
            font-size: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🛍️ Flipkart Review Sentiment Analyzer")

user_name = st.text_input("Your Name")

review_text = st.text_area(
    "Enter Product Review",
    height=150,
    placeholder="Type or paste a Flipkart product review here..."
)

if st.button("Analyze Sentiment"):
    if not review_text.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_text)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        name = user_name if user_name else "User"

        if prediction == "Positive":
            st.success(f"Hello {name}! Thank you for your positive feedback 😊")
        elif prediction == "Negative":
            st.error(f"Hello {name}! We’re sorry about your experience 😔")
        else:
            st.info(f"Hello {name}! Thanks for your balanced feedback 🙂")
