import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

DATA_PATH = Path(__file__).parent / "news_dataset.csv"

@st.cache_resource
def train_model():
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].astype(str)
    data["label"] = data["label"].astype(str).str.lower().str.strip()
    data = data[data["label"].isin(["real", "fake"])]

    if len(data) < 10 or data["label"].nunique() < 2:
        raise ValueError("Dataset must contain at least 10 rows and both 'real' and 'fake' labels.")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(data["text"], data["label"])
    return model, len(data)

st.title("📰 Fake News Detection")
st.write("Enter a headline or article and let a machine-learning model estimate whether its wording resembles examples labelled real or fake in the training dataset.")

st.info("Educational demo only: this model recognizes text patterns, not truth. Verify important claims with reliable sources.")

try:
    model, row_count = train_model()
    st.caption(f"Model trained on {row_count} example rows from news_dataset.csv.")
except Exception as error:
    st.error(f"Could not train the model: {error}")
    st.stop()

article = st.text_area(
    "Paste a news headline or article",
    placeholder="Type or paste the news text here...",
    height=220
)

if st.button("Analyze News", type="primary", use_container_width=True):
    if len(article.strip()) < 20:
        st.warning("Please enter at least 20 characters so the model has some text to analyze.")
    else:
        prediction = model.predict([article])[0]
        probabilities = model.predict_proba([article])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(prediction)]) * 100

        if prediction == "fake":
            st.error("Prediction: Likely Fake News")
        else:
            st.success("Prediction: Likely Real News")

        st.metric("Model confidence for this label", f"{confidence:.1f}%")
        st.caption("This percentage is the model's estimated confidence, not the probability that the article is factually true.")

        with st.expander("How should I interpret this result?"):
            st.write(
                "The model compares the words and word combinations in your text with patterns learned from the sample dataset. "
                "A wrong or biased training dataset can produce wrong predictions. Check the original source, publication date, "
                "author, and independent reporting before deciding whether a claim is reliable."
            )

st.divider()
st.subheader("How it works")
st.markdown("""
1. **TF-IDF** converts words and word combinations into numeric features.
2. **Logistic Regression** learns patterns from labelled examples.
3. The model returns the label it estimates is the closer match: `real` or `fake`.
""")
st.caption("Built with Python, scikit-learn, pandas, and Streamlit.")
