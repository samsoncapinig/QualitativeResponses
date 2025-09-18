"""
Streamlit app: Training Feedback Summarizer
File: streamlit_training_feedback_app.py

Features:
- Upload CSV/XLSX files.
- Detect qualitative/text columns automatically.
- Focused summarization on two guiding questions:
  1. What went well during the training?
  2. What can be improved in the conduct of training?
- Extract bullet-style themes and representative responses for each question.
- Export summary as TXT or CSV.

Dependencies:
- streamlit
- pandas
- scikit-learn

Install: pip install streamlit pandas scikit-learn
Run: streamlit run streamlit_training_feedback_app.py

"""

import streamlit as st
import pandas as pd
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Training Feedback Summarizer", layout="wide")

st.title("Training Feedback Summarizer — Streamlit")
st.markdown(
    "Upload a CSV or Excel file with qualitative responses (e.g. feedback). The app will focus on two guiding questions and generate summaries with themes and sample responses."
)


@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\-']", ' ', s)
    s = s.lower()
    return s


@st.cache_data
def extract_top_terms(series, n_terms=8, ngram_range=(1,2)):
    texts = series.dropna().astype(str).map(clean_text)
    texts = texts[texts.str.len() > 2]
    if len(texts) == 0:
        return []
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=ngram_range, max_features=2000)
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    ranked = sorted(zip(terms, sums), key=lambda x: x[1], reverse=True)
    top = [t for t, _ in ranked[:n_terms]]
    return top


def find_examples(series, term, n=3):
    texts = series.dropna().astype(str)
    pattern = re.compile(re.escape(term), flags=re.I)
    matches = [t for t in texts if pattern.search(t)]
    if len(matches) == 0:
        matches = [t for t in texts if len(str(t).strip()) > 3]
    return matches[:n]


# Sidebar: upload
st.sidebar.header("Upload & Settings")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"], accept_multiple_files=False)

if uploaded is not None:
    df = load_data(uploaded)
    if df is None:
        st.stop()

    st.sidebar.success(f"Loaded `{uploaded.name}` — {df.shape[0]} rows, {df.shape[1]} columns")

    # Ask user to select which columns correspond to the two guiding questions
    st.sidebar.markdown("### Map your columns to the guiding questions")
    cols = df.columns.tolist()
    col_well = st.sidebar.selectbox("Select column for 'What went well?'", options=[None]+cols, index=cols.index(cols[0]) if cols else 0)
    col_improve = st.sidebar.selectbox("Select column for 'What can be improved?'", options=[None]+cols, index=cols.index(cols[-1]) if cols else 0)

    n_terms = st.sidebar.slider("Top themes per question", min_value=3, max_value=12, value=6)
    n_examples = st.sidebar.slider("Examples per theme", min_value=1, max_value=5, value=2)

    st.subheader("Data preview")
    st.dataframe(df.head(50))

    st.subheader("Summarized Feedback")
    output_lines = []

    # Function to summarize a selected column
    def summarize_column(col, title):
        if col is None or col not in df.columns:
            st.warning(f"No column selected for: {title}")
            return
        st.markdown(f"## {title}")
        output_lines.append(f"## {title}\n")
        series = df[col].astype(str)
        nonempty = series.map(lambda x: str(x).strip()).replace({"nan":""})
        nonempty_count = sum(nonempty.map(lambda x: len(x) > 0))
        st.markdown(f"**Responses (non-empty):** {nonempty_count}")
        output_lines.append(f"Responses (non-empty): {nonempty_count}\n")
        top_terms = extract_top_terms(series, n_terms=n_terms)
        if not top_terms:
            st.write("No meaningful text found in this column.")
            output_lines.append("No meaningful text found.\n")
            return
        for i, term in enumerate(top_terms, start=1):
            examples = find_examples(series, term, n=n_examples)
            st.markdown(f"- **Theme {i}:** {term}")
            output_lines.append(f"- Theme {i}: {term}\n")
            if len(examples) > 0:
                st.markdown("  - Examples:")
                output_lines.append("  - Examples:\n")
                for ex in examples:
                    ex_short = ex if len(ex) <= 250 else ex[:247] + "..."
                    st.markdown(f"    - {ex_short}")
                    output_lines.append(f"    - {ex_short}\n")
        st.markdown("---")
        output_lines.append("\n")

    summarize_column(col_well, "What went well during the training?")
    summarize_column(col_improve, "What can be improved in the conduct of training?")

    summary_text = "\n".join(output_lines)

    st.subheader("Export")
    col1, col2 = st.columns([1,1])
    with col1:
        st.download_button("Download summary (TXT)", data=summary_text, file_name="training_feedback_summary.txt", mime="text/plain")
    with col2:
        csv_bytes = summary_text.encode('utf-8')
        st.download_button("Download summary (CSV)", data=csv_bytes, file_name="training_feedback_summary.csv", mime="text/csv")

    st.info("This app summarizes feedback specifically around what went well and what can be improved, using TF-IDF keyword ranking and representative examples.")

else:
    st.info("Upload a CSV or Excel file using the sidebar to begin.")
