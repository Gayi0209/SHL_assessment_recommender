import streamlit as st
import requests

API_URL = "https://YOUR-BACKEND.onrender.com/recommend"

st.set_page_config(page_title="SHL Assessment Recommender")

st.title("SHL Assessment Recommendation System")

query = st.text_area("Enter job description or hiring query")
top_k = st.slider("Top K Recommendations", 5, 10, 10)

if st.button("Recommend"):
    if not query.strip():
        st.warning("Please enter a query")
    else:
        with st.spinner("Fetching recommendations..."):
            response = requests.post(
                API_URL,
                json={"query": query, "top_k": top_k},
                timeout=60
            )

        if response.status_code == 200:
            for r in response.json():
                st.subheader(r["name"])
                st.markdown(f"[View Assessment]({r['url']})")
                st.write("Test Types:", ", ".join(r["test_type"]))
                st.divider()
        else:
            st.error("Backend error")
