import streamlit as st
import joblib
from pathlib import Path
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}

h1, h2, h3, p, label {
    color: #1f2937 !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: #1f2937 !important;
}

[data-testid="stTextArea"] textarea {
    background-color: white !important;
    color: #111827 !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

metrics = joblib.load(BASE_DIR / "model_metrics.pkl")
 #Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Load saved model
model = joblib.load(BASE_DIR / "spam_model.pkl")
vectorizer = joblib.load(BASE_DIR / "tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Smart Spam Detection",
    page_icon="🛡️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f0f4f8;
}
h1 {
    color: #17365D;
    text-align: center;
}
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Application heading
st.title("🛡️ Smart Spam Detection")
st.caption("AI-Powered SMS Classification System")

st.divider()

# Dashboard
col1, col2 = st.columns(2)

with col1:
    st.metric("Total SMS Dataset", "5,572")

with col2:
    st.metric("Categories", "2")

st.divider()

# SMS input
st.subheader("📩 Check Your SMS")

message = st.text_area(
    "Enter your SMS message:",
    placeholder="Type your message here..."
)

if st.button("Analyze Message"):
    if message.strip():

        message_tfidf = vectorizer.transform([message])
        prediction = model.predict(message_tfidf)
        st.session_state.history.append({
    "Message": message,
    "Result": prediction[0].upper()
})

        if prediction[0] == "spam":
            st.error("⚠️ SPAM MESSAGE DETECTED")
        else:
            st.success("✅ THIS IS A NORMAL MESSAGE (HAM)")

    else:
        st.warning("Please enter a message.")

st.divider()

st.caption("Developed using Python, Machine Learning and Streamlit")
st.divider()

st.subheader("📊 Prediction Dashboard")

history = st.session_state.history

total = len(history)

spam_count = sum(
    item["Result"] == "SPAM"
    for item in history
)

ham_count = sum(
    item["Result"] == "HAM"
    for item in history
)

# Display statistics
col1, col2, col3 = st.columns(3)

col1.metric("Total Checked", total)
col2.metric("Spam Messages", spam_count)
col3.metric("Normal Messages", ham_count)

# Display history
st.subheader("🕘 Message History")

if history:
    st.dataframe(
        history,
        use_container_width=True
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()

else:
    st.info("No messages checked yet.")
   # Model Performance Dashboard

st.divider()
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric(
    "Accuracy",
    f"{metrics['accuracy'] * 100:.2f}%"
)

col2.metric(
    "Precision",
    f"{metrics['precision'] * 100:.2f}%"
)

col3, col4 = st.columns(2)

col3.metric(
    "Recall",
    f"{metrics['recall'] * 100:.2f}%"
)

col4.metric(
    "F1 Score",
    f"{metrics['f1_score'] * 100:.2f}%"
)

st.subheader("Confusion Matrix")

st.write(metrics["confusion_matrix"])