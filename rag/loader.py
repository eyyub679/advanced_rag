from langchain_core.documents import Document
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def load():
    # convert genre ids to text
    id2text={}
    url=f"{os.environ["TMDB_URL"]}/genre/movie/list"
    params={"api_key":os.environ["TMDB_API_KEY"],"language":"en"}
    response=requests.get(url,params=params)
    for genre in response.json()["genres"]:
        id2text[genre["id"]]=genre["name"]

    # Define parameters for movie search
    movie_params={
        "api_key":os.environ["TMDB_API_KEY"],
        "with_genres":10749, # for romantic movies
        "sort_by":"vote_average.desc",
        "vote_count.gte":500,
        "page":None
    }
    # define parameters for recommendations
    recommendation_params={
        "api_key":os.environ["TMDB_API_KEY"],
        "sort_by":"vote_average.desc",
        "vote_count.gte":500,
        "page":1 # will only get the first 20 (at max) recommendations
    }
    # Url that will be used in get request to get movies
    url=f"{os.environ["TMDB_URL"]}/discover/movie"

    distinct_movies=set() # recommendations can overlap
    docs=[]

    for page in range(1,101):
        params=movie_params
        params["page"]=page
        response=requests.get(url,params=params)
        response.raise_for_status()
        results=response.json().get("results",[])
        if results:
            for movie in results:
                page_content=f"""
                    Movie name:\n{movie["title"]}\n
                    Release date:\n{movie["release_date"]}\n
                    Average vote:\n{movie["vote_average"]}/10\n
                    Overview:\n{movie["overview"]}\n
                    Recommendations:\n
                """
                metadata={
                    "movie_id":movie["id"],
                    "genres":[id2text[id] for id in movie["genre_ids"]]
                }
                recommendation_url=f"{os.environ["TMDB_URL"]}/movie/{movie["id"]}/recommendations"
                response=requests.get(recommendation_url,params=recommendation_params)
                recommendations=response.json().get("results",[])
                if recommendations:
                    for recommendation in recommendations:
                        if recommendation["id"] not in distinct_movies:
                            distinct_movies.add(recommendation["id"])
                            page_content+=f"\n{recommendation["title"]}\n"
                    doc=Document(
                        page_content=page_content.strip(),
                        metadata=metadata
                    )
                    docs.append(doc)
    return docs
