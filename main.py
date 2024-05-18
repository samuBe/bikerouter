import streamlit as st
import replicate

st.write('hello world')

if 'REPLICATE_API_TOKEN' in st.secrets:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']


input = {
    "prompt": "List 10 places to visit in 2024",
    "temperature": 0.2
}

output = replicate.run(
    "snowflake/snowflake-arctic-instruct",
    input=input
)
st.write("".join(output))
