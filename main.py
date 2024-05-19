import streamlit as st
from langchain.chains import LLMChain
import replicate
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
import os
from streamlit_searchbox import st_searchbox
from typing import List 
import requests
import uuid


# Put some nice things at the top
st.title('Bike router')
st.write('hello world')

mapbox_token = os.getenv('MAPBOX_TOKEN')
token = uuid.uuid4()

# function with list of labels
def search_city(searchterm: str) -> List[any]:
    if len(searchterm)<2:
        return []
    # sanitize!!!!!
    search_text = searchterm
    url = "https://api.mapbox.com/search/searchbox/v1/suggest"
    params = {"q": searchterm, "access_token": mapbox_token,"session_token": token, "types": "city"}
    res = requests.get(url, params=params)
    if res.status_code != 200:
       return [] 
    try:
        suggestions = res.json()['suggestions']
        results = list(map(lambda s: s['name'] + ', ' + s['place_formatted'],suggestions))
        return results
    except:
        return []
    return []

# pass search function to searchbox
city = st_searchbox(
    search_city,
    key="city",
)

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

