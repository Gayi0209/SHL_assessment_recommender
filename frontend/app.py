import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/recommend"

st.title("SHL Assessment Recommendation System")

query = st.text_area("Enter job description")

if st.button("Recommend"):
    res = requests.post(API_URL, json={"query": query})
    df = pd.DataFrame(res.json())
    st.dataframe(df[["name", "url", "test_type"]])
