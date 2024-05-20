# Welcome to BikeRouter!

## Description

This project combines the power of LLMs with location data and everyone's favourite computational problem: The Travelling Salesman problem!

* Enter the city you want to visit and you will get 10 must-visits in the city of choice! 
* Pick the locations you want to visit 
* Generate the shortest path between these locations

## Requirements

This project uses both Mapbox and Replicate. You will need to get API keys for both services:

* [Mapbox](https://www.mapbox.com/)
* [Replicate](https://replicate.com/)

## Running the project

Set-up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the necessary requirements

```bash
pip install -r requirements.txt
```

Create the API variables
```bash
export MAPBOX_TOKEN=
export REPLICATE_API_TOKEN=
```

Run the project
```bash
python -m streamlit run main.py
```

## Deploying the project

You can deploy BikeRouter using a Docker container. For this, the Dockerfile in the project has been added. You can use AWS ECS or alternatives by Azure and GCP.

I have deployed BikeRouter on Railway: [here](https://bikerouter.samuelberton.com)

## Next steps

* Add some context, i.e., who is doing the bike trip
* Clean up the code, for now all the code is in the same file. In the future, make things more modular
* Write tests to see if everything is working as expected. Here, a framework like PyTest can be leveraged
