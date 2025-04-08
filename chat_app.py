
import streamlit as st
import pandas as pd
import re
from model import hybrid_recommendation
from transformers import pipeline

st.set_page_config(page_title="HR Recommender Chatbot", layout="wide")
st.title("🤖 HR Talent Recommender Chatbot")

# Load or upload dataset
uploaded_file = st.file_uploader("Upload employee dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully.")
else:
    df = pd.read_csv("Diverse_HR_Employee_Dataset_200.csv")
    st.info("Using default dataset.")

# Load LLM for query rewriting
#@st.cache_resource
#def load_llm():
#    return pipeline("text2text-generation", model="google/flan-t5-base", use_auth_token="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")

#llm = load_llm()

# Extract top_k using regex
def extract_top_k(query):
    match = re.search(r"(\d+)\s+(?:candidates|profiles|people)?", query.lower())
    return int(match.group(1)) if match else 5

# Rewrite vague HR query using LLM
def rewrite_query_with_llm(raw_query, history=[]):
    return raw_query

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = pd.DataFrame()

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
query = st.chat_input("Type your query here (e.g., 'Give me 5 Azure-certified Data Scientists')")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append(query)

    structured_query = rewrite_query_with_llm(query, st.session_state.history)
    top_k = extract_top_k(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            try:
                results = hybrid_recommendation(df, structured_query, top_k=top_k)
                st.session_state.last_results = results
                if results.empty:
                    st.markdown("❌ No matching candidates found.")
                    response = "No results found for your query."
                else:
                    response_lines = ["Here are the top recommended candidates:"]
                    for _, row in results.iterrows():
                        response_lines.append(f"- {row['Employee Name']} ({row['Job Title']}, {row['Location']})")
                    response = "\n".join(response_lines)
                    st.markdown(response)
            except Exception as e:
                response = f"⚠️ Error: {e}"
                st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# Export results as CSV
if not st.session_state.last_results.empty:
    st.download_button("⬇️ Download Results as CSV", 
                       data=st.session_state.last_results.to_csv(index=False),
                       file_name="recommended_employees.csv",
                       mime="text/csv")
