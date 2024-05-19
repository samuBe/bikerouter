import streamlit as st
from langchain.chains import LLMChain
import replicate
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
import os

# Put some nice things at the top
st.title('Bike router')
st.write('hello world')

city = st.text_input("Enter the city you want to visit", key="city")


@st.cache_resource
def getChain():

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    llm = Replicate(
               model="snowflake/snowflake-arctic-instruct",
               model_kwargs={'temperature':0}
    )


    prompt = PromptTemplate(
        template="""Return a comma-separated list of the 10 best landmarks in {city}. Only return the list
        {format_instructions}
            """,
        input_variables=["city"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser
    return chain


# Run the llm
chain = getChain()
if city and len(city)>0:
    output = chain.invoke({"city": city})
    print(output)
    st.write(output)

