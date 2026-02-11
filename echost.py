import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


# ======================================================
# CONFIG
# ======================================================
st.set_page_config(page_title="Sentiment Analytics Dashboard", layout="wide")

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ======================================================
# TEXT CLEANING (IMPORTANT: MUST MATCH TRAINING)
# ======================================================
def clean_text(text):

    text = str(text).lower().strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters & numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text


# ======================================================
# LOAD MODEL & TOKENIZER
# ======================================================
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model("lstm_sentiment_model.keras")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Try to auto-detect max length
    try:
        max_len = model.input_shape[1]
    except:
        max_len = 120

    return model, tokenizer, max_len


model, tokenizer, MAX_LEN = load_model()


# ======================================================
# DEBUG INFO (SIDEBAR)
# ======================================================
with st.sidebar.expander("🛠 Debug Info"):

    st.write("Tokenizer vocabulary size:", len(tokenizer.word_index))
    st.write("Model input max length:", MAX_LEN)

    if len(tokenizer.word_index) < 1000:
        st.warning("⚠️ Tokenizer seems too small. Possible mismatch.")


# ======================================================
# USER INPUT: LIVE SENTIMENT PREDICTION
# ======================================================
st.markdown("## ✍️ Analyze Your Own Review")

user_review = st.text_area(
    "Enter a review to analyze sentiment:",
    height=120,
    placeholder="Type your review here..."
)

if st.button("Predict Sentiment"):

    if user_review.strip() == "":

        st.warning("⚠️ Please enter some text first.")

    else:

        # Clean text
        clean_review = clean_text(user_review)

        # Tokenize
        seq = tokenizer.texts_to_sequences([clean_review])

        # Check tokenizer
        if len(seq[0]) == 0:

            st.error("""
⚠️ No words recognized by tokenizer.

Possible reasons:
• Wrong tokenizer file
• Different preprocessing
• Model/tokenizer mismatch
""")

        else:

            pad = pad_sequences(
                seq,
                maxlen=MAX_LEN,
                padding="post",
                truncating="post"
            )

            probs = model.predict(pad, verbose=0)[0]

            pred_class = np.argmax(probs)

            sentiment = LABELS[pred_class]

            confidence = np.max(probs) * 100


            # Show result
            st.success(f"### 🧠 Predicted Sentiment: **{sentiment}**")
            st.info(f"Confidence: {confidence:.2f}%")


            # Show probability breakdown
            st.markdown("### 📊 Prediction Probabilities")

            prob_df = pd.DataFrame({
                "Sentiment": ["Negative", "Neutral", "Positive"],
                "Probability (%)": np.round(probs * 100, 2)
            })

            st.bar_chart(prob_df.set_index("Sentiment"))


st.divider()


# ======================================================
# LOAD DATASET
# ======================================================
@st.cache_data
def load_data():

    df = pd.read_excel("chatgpt_style_reviews_dataset (1).xlsx")

    df["review"] = df["review"].fillna("").apply(clean_text)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


df = load_data()


# ======================================================
# BULK SENTIMENT PREDICTION
# ======================================================
def predict_sentiment(texts):

    texts = [clean_text(t) for t in texts]

    seq = tokenizer.texts_to_sequences(texts)

    pad = pad_sequences(
        seq,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probs = model.predict(pad, verbose=0)

    preds = np.argmax(probs, axis=1)

    return [LABELS[i] for i in preds]


if "Predicted_Sentiment" not in df.columns:

    df["Predicted_Sentiment"] = predict_sentiment(
        df["review"].tolist()
    )


# ======================================================
# SIDEBAR MENU
# ======================================================
st.sidebar.title("📊 Analysis Menu")

menu = st.sidebar.selectbox(
    "Select Question",
    [
        "1. Overall Sentiment",
        "2. Sentiment vs Rating",
        "3. Keywords by Sentiment",
        "4. Sentiment Over Time",
        "5. Verified vs Non-Verified",
        "6. Review Length Analysis",
        "7. Sentiment by Location",
        "8. Sentiment by Platform",
        "9. Sentiment by Version",
        "10. Negative Feedback Themes"
    ]
)


# ======================================================
# 1. OVERALL SENTIMENT
# ======================================================
if menu == "1. Overall Sentiment":

    st.title("1️⃣ Overall Sentiment Distribution")

    counts = df["Predicted_Sentiment"].value_counts()

    percent = counts / counts.sum() * 100

    st.dataframe(pd.DataFrame({
        "Sentiment": counts.index,
        "Count": counts.values,
        "Percentage (%)": percent.round(2)
    }))


    fig, ax = plt.subplots()

    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    st.pyplot(fig)


# ======================================================
# 2. SENTIMENT vs RATING
# ======================================================
elif menu == "2. Sentiment vs Rating":

    st.title("2️⃣ Sentiment vs Rating")

    table = pd.crosstab(
        df["rating"],
        df["Predicted_Sentiment"]
    )

    st.dataframe(table)


    fig, ax = plt.subplots()

    sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)


# ======================================================
# 3. KEYWORDS BY SENTIMENT
# ======================================================
elif menu == "3. Keywords by Sentiment":

    st.title("3️⃣ Keywords by Sentiment")

    sentiment = st.selectbox(
        "Choose Sentiment",
        df["Predicted_Sentiment"].unique()
    )

    text = " ".join(
        df[df["Predicted_Sentiment"] == sentiment]["review"]
    )


    wc = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)


    fig, ax = plt.subplots(figsize=(10, 4))

    ax.imshow(wc)
    ax.axis("off")

    st.pyplot(fig)


# ======================================================
# 4. SENTIMENT OVER TIME
# ======================================================
elif menu == "4. Sentiment Over Time":

    st.title("4️⃣ Sentiment Over Time")

    df_time = df.dropna(subset=["date"]).copy()

    df_time["Month"] = df_time["date"].dt.to_period("M").astype(str)

    trend = pd.crosstab(
        df_time["Month"],
        df_time["Predicted_Sentiment"]
    )

    st.line_chart(trend)


# ======================================================
# 5. VERIFIED vs NON-VERIFIED
# ======================================================
elif menu == "5. Verified vs Non-Verified":

    st.title("5️⃣ Verified vs Non-Verified Users")

    table = pd.crosstab(
        df["verified_purchase"],
        df["Predicted_Sentiment"]
    )

    st.dataframe(table)

    st.bar_chart(table)


# ======================================================
# 6. REVIEW LENGTH
# ======================================================
elif menu == "6. Review Length Analysis":

    st.title("6️⃣ Review Length Analysis")

    df["review_length"] = df["review"].str.len()

    avg_len = df.groupby(
        "Predicted_Sentiment"
    )["review_length"].mean()

    st.bar_chart(avg_len)


# ======================================================
# 7. LOCATION
# ======================================================
elif menu == "7. Sentiment by Location":

    st.title("7️⃣ Sentiment by Location")

    loc = pd.crosstab(
        df["location"],
        df["Predicted_Sentiment"]
    ).sort_values(
        by="Positive",
        ascending=False
    ).head(10)

    st.dataframe(loc)


# ======================================================
# 8. PLATFORM
# ======================================================
elif menu == "8. Sentiment by Platform":

    st.title("8️⃣ Sentiment by Platform")

    platform = pd.crosstab(
        df["platform"],
        df["Predicted_Sentiment"]
    )

    st.bar_chart(platform)


# ======================================================
# 9. VERSION
# ======================================================
elif menu == "9. Sentiment by Version":

    st.title("9️⃣ Sentiment by Version")

    version = pd.crosstab(
        df["version"],
        df["Predicted_Sentiment"]
    )

    st.line_chart(version)


# ======================================================
# 10. NEGATIVE THEMES
# ======================================================
elif menu == "10. Negative Feedback Themes":

    st.title("🔟 Negative Feedback Themes")

    neg_reviews = df[
        df["Predicted_Sentiment"] == "Negative"
    ]["review"]


    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=20
    )


    X = vectorizer.fit_transform(neg_reviews)


    keywords = pd.DataFrame({
        "Keyword": vectorizer.get_feature_names_out(),
        "Frequency": X.sum(axis=0).A1
    }).sort_values(
        by="Frequency",
        ascending=False
    )


    st.dataframe(keywords)
