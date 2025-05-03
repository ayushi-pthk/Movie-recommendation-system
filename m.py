import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
movie_ids_titles = pd.read_csv("movies.csv")
movie_ids_ratings = pd.read_csv("ratings.csv")

# Select necessary columns
movie_ids_titles = movie_ids_titles[['movieId', 'title']]
movie_ids_ratings = movie_ids_ratings[['userId', 'movieId', 'rating']]

# Merge dataframes
merged_movie_df = pd.merge(movie_ids_ratings, movie_ids_titles, on='movieId')

# Calculate correlations
all_movie_correlations = merged_movie_df.groupby('title')['rating'].apply(lambda x: x.corr(merged_movie_df['rating']))

# Load test data
test_movies = pd.DataFrame([['Forrest Gump (1994)', 4.0], ['Fight Club (1999)', 3.5], ['Interstellar (2014)', 4.0]], columns=['Movie_Name', 'Movie_Rating'])

# Calculate recommendations
recommended_movies = all_movie_correlations.loc[test_movies['Movie_Name']].dropna()
recommended_movies = recommended_movies.map(lambda movie_corr: movie_corr * test_movies["Movie_Rating"])
recommended_movies = recommended_movies.sort_values(ascending=False)

# Print recommendations
print("Movies related to Forest Gump:")
print(recommended_movies.head(10))
