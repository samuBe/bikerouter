import streamlit as st
import replicate
import os

st.write('hello world')

input = {
    "prompt": "List 10 places to visit in 2024",
    "temperature": 0.2
}

output = replicate.run(
    "snowflake/snowflake-arctic-instruct",
    input=input
)
st.write("".join(output))
