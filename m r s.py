import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import dask.dataframe as dd

# Load data in chunks using Dask
movie_metadata = dd.read_csv("movies.csv", blocksize=10 ** 6)
movie_ratings = dd.read_csv("ratings.csv", blocksize=10 ** 6)

# Preprocess data using vectorized operations
movie_metadata = movie_metadata.drop("genres", axis=1)
movie_ratings = movie_ratings.drop("timestamp", axis=1)
merged_data = dd.merge(movie_ratings, movie_metadata, on="movieId")

# Compute the merged data
merged_data = merged_data.compute()

# Analyze data using vectorized operations
rating_mean = merged_data.groupby("title")["rating"].mean()
rating_count = merged_data.groupby("title")["rating"].count()
movie_rating_mean_count = pd.DataFrame({"rating_mean": rating_mean, "rating_count": rating_count})

# Visualize data using vectorized operations
plt.figure(figsize=(10, 8))
sns.set_style("darkgrid")
movie_rating_mean_count["rating_mean"].hist(bins=30, color="purple")
plt.show()

# Recommendation algorithm using parallel processing
def recommend_movies(merged_data, test_movies):
    all_movie_correlations = merged_data.pivot_table(index="userId", columns="title", values="rating").corr(method="pearson", min_periods=50)
    recommended_movies_list = Parallel(n_jobs=-1)(delayed(recommend_movie)(all_movie_correlations, test_movies.iloc[i]) for i in range(len(test_movies)))
    recommended_movies = pd.concat(recommended_movies_list)
    recommended_movies.sort_values(inplace=True, ascending=False)
    return recommended_movies

def recommend_movie(all_movie_correlations, test_movie):
    movie_name, movie_rating = test_movie
    movie = all_movie_correlations[movie_name].dropna()
    movie = movie.map(lambda movie_corr: movie_corr * movie_rating)
    return movie.to_frame(name="recommendation").reset_index()

test_movies = pd.DataFrame([["Forrest Gump (1994)", 4.0], ["Fight Club (1999)", 3.5]], columns=["Movie_Name", "Movie_Rating"])
recommended_movies = recommend_movies(merged_data, test_movies)
print("Movies related to Forest Gump:")
print(recommended_movies.head(10))
