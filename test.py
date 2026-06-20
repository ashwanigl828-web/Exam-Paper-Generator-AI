import streamlit as st
try:
    s = st.secrets["FOO"]
except Exception:
    pass

st.set_page_config(page_title="Test")
st.write("Hello")
