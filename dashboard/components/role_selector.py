# role_selector.py (Optional)
import streamlit as st

def get_user_role():
    st.sidebar.header("User Settings")
    return st.sidebar.selectbox("Select Your Role", ["Patient", "Doctor", "Clinician"])