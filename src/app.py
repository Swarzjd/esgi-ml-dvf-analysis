import streamlit as st

st.set_page_config(page_title="Analyse DVF", page_icon="🏠", layout="wide")

navigation = st.navigation([
    st.Page("views/classification.py", title="Appartement ou maison ?", icon="🏠"),
    st.Page("views/estimation.py", title="Estimation du prix", icon="💶"),
])
navigation.run()
