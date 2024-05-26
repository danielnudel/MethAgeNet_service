import streamlit as st

st.header("Choose your option from the left panel.")

def menu():
    st.sidebar.page_link("upload_single_hist.py", label="Single hist")
    st.sidebar.page_link("upload_multiple_hists_for_single_sample.py", label="Multiple hists hists for same donor")