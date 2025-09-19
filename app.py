import streamlit as st
import pandas as pd
from collections import Counter
import re

st.title("Training Feedback Summary App")

st.write("""
Upload your CSV file(s) containing qualitative responses.  
The app will summarize the most common responses into two categories:  
1. Positive feedback  
2. Areas for improvement
""")

uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    return text.strip()

if uploaded_files:
    dataframes = [pd.read_csv(file) for file in uploaded_files]
    df = pd.concat(dataframes, ignore_index=True)

    # Extract last 3 columns (Insights, Learnings, For improvement)
    qual_cols = df.iloc[:, -3:]
    qual_cols.columns = ["Insights", "Learnings", "For_Improvement"]

    # Combine positive feedback (Insights + Learnings)
    positive_feedback = qual_cols["Insights"].dropna().tolist() + qual_cols["Learnings"].dropna().tolist()
    improvement_feedback = qual_cols["For_Improvement"].dropna().tolist()

    # Preprocess
    positive_feedback_clean = [preprocess_text(x) for x in positive_feedback if x not in ["none", "n/a", ""]]
    improvement_feedback_clean = [preprocess_text(x) for x in improvement_feedback if x not in ["none", "n/a", ""]]

    # Count most common phrases/words
    positive_counter = Counter(positive_feedback_clean)
    improvement_counter = Counter(improvement_feedback_clean)

    st.subheader("✅ Positive Feedback (Prevailing Responses)")
    if positive_feedback_clean:
        positive_summary = "\n".join([f"- {item} ({count} mentions)" for item, count in positive_counter.most_common(10)])
        st.code(positive_summary, language="text")  # copy button enabled
    else:
        st.write("No positive feedback provided.")

    st.subheader("⚠️ Areas for Improvement (Prevailing Responses)")
