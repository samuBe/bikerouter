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
        results = list(map(lambda s: (s['name'] + ', ' + s['place_formatted'], s['mapbox_id'] ),suggestions))
        return results
    except:
        return []
    return []

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
def get_chain():

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    llm = Replicate(
               model="snowflake/snowflake-arctic-instruct",
               model_kwargs={'temperature':0}
    )


    prompt = PromptTemplate(
        template="""Return a comma-separated list of at least 10 of the best landmarks in {city}. Only return the list
        {format_instructions}
            """,
        input_variables=["city"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser
    return chain

def get_landmark_locations(landmarks, long, lat):
    data = []
    for lm in landmarks:
        name = lm
        features = retrieve_landmark(lm, f"{long},{lat}")
        coor = features['geometry']['coordinates']
        long, lat = coor
        data.append([name, long, lat])

    # put into a pandas dataframe
    df = pd.DataFrame(data=data, columns=['name', 'longitude', 'latitude'])
    print(df)
    return df


# Run the llm
chain = get_chain()
if city_id and len(city_id)>0:
    city = retrieve_city(city_id)
    coor = city['geometry']['coordinates']
    long, lat = coor
    landmarks = chain.invoke({"city": city['properties']['full_address'] })
    landmark_locations = get_landmark_locations(landmarks, long, lat)
    st.map(landmark_locations)
    st.write(landmarks)

