import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from nltk.stem import PorterStemmer


def read_api_key(filename):
    with open(filename, "r") as file:
        api_key = file.read().strip()
    return api_key


API_KEY_FILE = "hey.txt"
API_KEY = read_api_key(API_KEY_FILE)


# Load data
def load_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    return movies, credits


# Data preprocessing for content-based
def preprocess_content_based(movies, credits):
    # Merge datasets
    movies = movies.merge(credits, on="title")

    # Select relevant columns
    movies = movies[
        [
            "movie_id",
            "title",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "popularity",
        ]
    ]

    # Drop missing values
    movies.dropna(inplace=True)

    # Extract and clean data
    movies["genres"] = movies["genres"].apply(lambda x: extract_names(x, "name"))
    movies["keywords"] = movies["keywords"].apply(lambda x: extract_names(x, "name"))
    movies["cast"] = movies["cast"].apply(lambda x: extract_cast(x))
    movies["crew"] = movies["crew"].apply(lambda x: extract_directors(x))

    # Clean text data
    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    movies["tags"] = movies.apply(create_tags, axis=1)

    return movies


# Helper functions for data extraction
def extract_names(data, key):
    try:
        data = ast.literal_eval(data)
        return [item[key].replace(" ", "") for item in data]
    except (ValueError, KeyError, SyntaxError):
        return []


def extract_cast(cast_data):
    try:
        cast_data = ast.literal_eval(cast_data)
        return [member["name"].replace(" ", "") for member in cast_data[:3]]
    except (ValueError, KeyError, SyntaxError):
        return []


def extract_directors(crew_data):
    directors = []
    try:
        crew_data = ast.literal_eval(crew_data)
        for member in crew_data:
            if member["job"] == "Director":
                directors.append(member["name"].replace(" ", ""))
    except (ValueError, KeyError, SyntaxError):
        pass
    return directors


# Function to create tags
def create_tags(row):
    tags = row["overview"] + row["genres"] + row["keywords"] + row["cast"] + row["crew"]
    return " ".join(tags).lower()


# Stemming function
def stem_text(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(word) for word in text.split()])


# Vectorization and similarity computation
def compute_similarity(movies):
    # Stem tags
    movies["tags"] = movies["tags"].apply(stem_text)

    # Vectorize tags
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"]).toarray()

    # Compute cosine similarity matrix
    sim = cosine_similarity(vectors)

    return sim


# Recommendation function for content-based
def recommend_content_based(movie_title, movies, sim):
    try:
        # Get movie index
        idx = movies[movies["title"] == movie_title].index[0]

        # Calculate similarity scores
        sim_scores = list(enumerate(sim[idx]))

        # Sort movies based on similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get top recommendations
        top_movies_indices = [i[0] for i in sim_scores[1:4]]
        recommended_movies = movies.iloc[top_movies_indices]["title"].tolist()

        return recommended_movies

    except IndexError:
        return []


# Function to fetch movie poster URL and TMDb page URL
def fetch_poster_and_url(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(url)
    data = response.json()
    if data["results"]:
        poster_path = data["results"][0]["poster_path"]
        movie_id = data["results"][0]["id"]
        full_poster_path = f"https://image.tmdb.org/t/p/w500{poster_path}"
        full_movie_url = f"https://www.themoviedb.org/movie/{movie_id}"
        return full_poster_path, full_movie_url
    return "", ""


# Popularity-based recommendation function
def recommend_popularity_based(movies):
    popular_movies = movies.sort_values("popularity", ascending=False).head(10)
    return popular_movies["title"].tolist()


# Main function to load data and perform recommendations
def main():
    movies, credits = load_data()
    movies = preprocess_content_based(movies, credits)
    sim = compute_similarity(movies)

    return movies, sim


if __name__ == "__main__":
    main()
