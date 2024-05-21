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
import markdown
from geopy import distance


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
    params = {"access_token": mapbox_token, "q": name, "proximity": proximity, 'types': 'poi', 'poi_category': 'tourist_attraction,museum,monument,historic,park,church,place of workship'}
    res = requests.get(url, params=params)
    if res.status_code != 200:
        return []
    try:
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
def get_llm(temperature=0, max_tokens=512, top_k=50):
    llm = Replicate(
               model="snowflake/snowflake-arctic-instruct",
               model_kwargs={'temperature':temperature, 'max_new_tokens': max_tokens, 'top_k': top_k }
    )
    return llm


@st.cache_resource
def get_landmark_chain():

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    llm = get_llm()

    prompt = PromptTemplate(
        template="""Return a comma-separated list of the 10 best landmarks in {city}. Only return the list
        {format_instructions}
            """,
        input_variables=["city"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser
    return chain

@st.cache_data
def get_landmark_locations(landmarks, long_city, lat_city):
    data = []
    for lm in landmarks:
        name = lm
        print(lm)
        features = retrieve_landmark(lm, f"{long_city},{lat_city}")
        # if mapbox does not find anything, continue
        if len(features)==0:
            continue
        coor = features['geometry']['coordinates']
        long, lat = coor
        # Check if the distance is unreasonable
        dist= distance.distance((lat_city, long_city), (lat, long)).km
        if dist>10:
            continue
        data.append([name, long, lat, True])

    # put into a pandas dataframe
    df = pd.DataFrame(data=data, columns=['Name', 'longitude', 'latitude', 'Include'])
    return df

@st.cache_resource
def get_fallback_chain():

    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    
    llm = get_llm()

    prompt = PromptTemplate(
        template="""You were asked to make a list of landmarks in {city}, can you format {output_string} as a comma-separated list. Don't return any introduction, just the list.
        {format_instructions}
            """,
        input_variables=["city", "output_string"],
        partial_variables={"format_instructions": format_instructions},
    )

    chain = prompt | llm | output_parser

    return chain

@st.cache_data
def run_llm(parameters):
    chain = get_landmark_chain()
    print(chain)
    out = chain.invoke(parameters)
    print('hello')
    if len(out) != 10:
    # TODO add fallback if not good
        out_string = ' '.join(out)
        parameters["output_string"] = out_string
        fallback = get_fallback_chain()
        out = fallback.invoke(parameters)
        return out
    return out

@st.cache_data
def tsp(chosen_landmarks):
    profile = "mapbox/cycling"
    coordinates = ";".join([f"{row['longitude']},{row['latitude']}" for index, row in chosen_landmarks.iterrows()])
    link = f"https://api.mapbox.com/optimized-trips/v1/{profile}/{coordinates}"
    params = {"access_token": mapbox_token}
    res = requests.get(link, params=params)
    if res.status_code != 200:
        return []
    try:
        return res.json()
    except:
        return []
    return []

@st.cache_data
def create_route(city, landmarks):
    llm = get_llm(0.2, 2048, 15)
    prompt = PromptTemplate(
        template="""You are an experienced tour guide in {city}. You love telling more about landmarks in a short way. Create a bike route for {city} in markdown, using headings with ##, passing by the following landmarks: 
        {landmarks}. 
        The route should be clearly connected through the text. End with the introduction of the next landmark {end}, as if it was the next destination, but don't discuss it.
        An excerpt for Paris: 
        ## Place d'Alma
        Our first stop is Place d'Alma with a superb view of the **Eiffel Tower**. The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
        It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the centerpiece of the 1889 World's Fair. 
        Although initially criticised by some of France's leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world.
        Next step, the **Arc the Triomphe**...
        """, 
       input_variables=["landmarks", "city", "end"]
    )
    chain = prompt | llm
    landmarks_string = "\n".join([f"{row['Name']}" for index, row in landmarks.iloc[:5,:].iterrows()])
    print(landmarks_string)
    part_one = chain.invoke({'city': city, 'landmarks':landmarks_string, 'end': landmarks.iloc[5,:]['Name']})
    if len(landmarks)<5:
        return part_one
    prompt = PromptTemplate(
        template="""You are an experienced tour guide in {city}. You love telling more about landmarks in a short way. Create a bike route for {city} in markdown, using headings with ##, passing by the following landmarks: 
        {landmarks}. 
        The route should be clearly connected through the text. Start your explanation with Continuing from {previous}
        An excerpt for Paris: 
        ## Place d'Alma
        Our first stop is Place d'Alma with a superb view of the **Eiffel Tower**. The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
        It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the centerpiece of the 1889 World's Fair. 
        Although initially criticised by some of France's leading artists and intellectuals for its design, it has since become a global cultural icon of France and one of the most recognisable structures in the world.
        Next step, the **Arc the Triomphe**...
        """, 
        input_variables=["landmarks", "city", "previous"]
    )
    chain = prompt | llm
    landmarks_string = "\n".join([f"{row['Name']}" for index, row in landmarks.iloc[5:,:].iterrows()])
    part_two = chain.invoke({'city': city, 'landmarks':landmarks_string, 'previous': landmarks.iloc[4,:]['Name']})
    return part_one + " " + part_two




@st.cache_data
def to_html(data, filename='BikeRouter-route'):
    return markdown.markdown(data)

if 'route' not in st.session_state:
    st.session_state.route = None

def generate_route(city, stops):
    route = create_route(city['properties']['full_address'], stops)
    st.session_state.route = route


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
    chosen_landmarks = user_input[user_input['Include']]
    st.map(chosen_landmarks)
    output = tsp(chosen_landmarks)

    distance = output['trips'][0]['distance']
    st.write(f"Total distance: {distance}m")

    waypoints = map(lambda wp: wp['waypoint_index'],output['waypoints'])
    stops  = chosen_landmarks.iloc[waypoints, :]

    st.button('Generate route', on_click=lambda : generate_route(city, stops))
    

route = st.session_state.route
if route and len(route)>0:
    st.markdown(route)

    route_html = to_html(route)

    st.download_button(
            label='Download the bike route!',
            data=route_html,
            file_name='BikeRouter-route.html',
            mime='text/html'
    )

