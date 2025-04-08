
import streamlit as st
import pandas as pd
from model import hybrid_recommendation

st.set_page_config(page_title="HR Talent Recommender", layout="wide")

st.title("HR Talent Recommendation System")
st.write("Search for top employee matches based on a natural language query.")

uploaded_file = st.file_uploader("Upload Employee Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Diverse_HR_Employee_Dataset_200.csv")
    st.info("Using default dataset (200 synthetic employees).")

query = st.text_input("Enter your HR Query (e.g. 'Looking for AI engineers with NLP'):")
location = st.text_input("Optional: Enter location to filter (e.g. Bangalore):").strip().lower()

top_k = st.slider("How many top candidates do you want?", min_value=1, max_value=20, value=5)

# Add location to query if provided
if location:
    query += f" in {location}"

if st.button("Run Recommendation") and query:
    with st.spinner("Finding top candidates..."):
        results = hybrid_recommendation(df, query, top_k=top_k)
        st.success("Top recommended employees:")
        st.dataframe(results)
