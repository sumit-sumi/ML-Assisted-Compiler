import streamlit as st
from compiler.lexer import tokenize
from compiler.error_checker import check_errors

st.title("ML-Assisted Compiler Prototype")
code = st.text_area("Enter Code")

if st.button("Analyze"):
    tokens = tokenize(code)
    result = check_errors(code)
    st.write("**Tokens:**", tokens)
    st.write("**ML Evaluation:**", result)
