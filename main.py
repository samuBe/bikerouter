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
import pandas as pd


# Put some nice things at the top
st.title('BikeRouter')
st.markdown("Made by [Samuel Berton](https://samuelberton.com) using [Streamlit](https://streamlit.io/)")

with open('description.md') as description:
    st.markdown(description.read())

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
        results = list(map(lambda s: (s['name'] + ', ' + s['place_formatted'], s['mapbox_id'] ),suggestions))
        return results
    except:
        return []
    return []

@st.cache_data
def retrieve_city(id):
    url = f"https://api.mapbox.com/search/searchbox/v1/retrieve/{id}"
    params = {"access_token": mapbox_token,"session_token": token}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        return []
    try:
        return res.json()['features'][0]
    except:
        return []
    return []

@st.cache_data
def retrieve_landmark(name, proximity):
    url = "https://api.mapbox.com/search/searchbox/v1/forward"
    params = {"access_token": mapbox_token, "q": name, "proximity": proximity, 'types': 'poi', 'poi_category': 'tourist_attraction,museum,monument,historic'}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        return []
    try:
        print(res.text)
        return res.json()['features'][0]
    except:
        return []
    return []


# pass search function to searchbox
city_id = st_searchbox(
    search_city,
    key="city",
)

@st.cache_resource
def get_llm():
    llm = Replicate(
               model="snowflake/snowflake-arctic-instruct",
               model_kwargs={'temperature':0}
    )
    return llm


@st.cache_resource
def get_landmark_chain():

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    llm = get_llm()

    prompt = PromptTemplate(
        template="""Return a comma-separated list of at least 10 of the best landmarks in {city}. Only return the list
        {format_instructions}
            """,
        input_variables=["city"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser
    return chain

@st.cache_data
def get_landmark_locations(landmarks, long, lat):
    data = []
    for lm in landmarks:
        name = lm
        features = retrieve_landmark(lm, f"{long},{lat}")
        coor = features['geometry']['coordinates']
        long, lat = coor
        data.append([name, long, lat, True])

    # put into a pandas dataframe
    df = pd.DataFrame(data=data, columns=['Name', 'longitude', 'latitude', 'Include'])
    print(df)
    return df

@st.cache_data
def run_llm(parameters):
    chain = get_landmark_chain()
    return chain.invoke(parameters)


# Run the llm
chain = get_landmark_chain()
if city_id and len(city_id)>0:
    city = retrieve_city(city_id)
    coor = city['geometry']['coordinates']
    long, lat = coor
    landmarks = run_llm({"city": city['properties']['full_address'] })
    st.session_state.landmark_locations = get_landmark_locations(landmarks, long, lat)
    landmark_locations = st.session_state.landmark_locations
    user_input = st.data_editor(landmark_locations, hide_index=True, disabled=('name', 'longitude', 'latitude'), 
                   column_config= {'longitude': None, 'latitude': None}, key='user_input', use_container_width=True)
    st.map(user_input[user_input['Include']])


