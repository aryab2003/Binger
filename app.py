import streamlit as st
import pandas as pd
from model import (
    main,
    recommend_content_based,
    recommend_popularity_based,
    fetch_poster_and_url,
)
import streamlit.components.v1 as components

# Load data and compute similarity
movies, sim = main()

# Streamlit app
st.title("Movie Recommender System")
st.markdown(
    """
Welcome to the Movie Recommender System! Select a movie from the dropdown menu 
and click the 'Recommend' button to find similar movies.
"""
)

# Dropdown for selecting recommendation type
rec_type = st.selectbox(
    "Select recommendation type", ["Content-Based", "Popularity-Based"]
)

# Dropdown for selecting movie (for content-based)
if rec_type == "Content-Based":
    selected_movie = st.selectbox("Select a movie", movies["title"].values)

# Button for recommendation
if st.button("Recommend"):
    if rec_type == "Content-Based":
        recommendations = recommend_content_based(selected_movie, movies, sim)
    else:
        recommendations = recommend_popularity_based(movies)

    # Display recommendations in beautiful cards
    st.subheader("Recommended Movies:")
    if recommendations:
        for movie in recommendations:
            poster_url, movie_url = fetch_poster_and_url(movie)
            card_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center; margin: 10px;">
                <a href="{movie_url}" target="_blank" style="text-decoration: none; color: inherit;">
                    <div style="background-color: #f9f9f9; border-radius: 15px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); transition: 0.3s; width: 200px; text-align: center; padding: 10px;">
                        <img src="{poster_url}" alt="{movie} poster" style="border-radius: 15px; width: 100%; height: auto;">
                        <h3 style="font-family: 'Arial', sans-serif;">{movie}</h3>
                    </div>
                </a>
            </div>
            """
            components.html(card_html, height=550)
    else:
        st.write("No recommendations found.")
