import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    movie_ids_titles=pd.read_csv(r"C:\Users\Pragya\Desktop\movie recommendation system\movies.csv")

    movie_ids_titles.head()

    movie_ids_titles.shape

    movie_ids_ratings=pd.read_csv(r"C:\Users\Pragya\Desktop\movie recommendation system\ratings.csv")

    movie_ids_ratings.head()

    movie_ids_ratings.shape

    movie_ids_titles.drop(['genres'],inplace=True,axis=1)

    movie_ids_titles.head()

    movie_ids_ratings.drop(["timestamp"],inplace=True,axis=1)

    movie_ids_ratings.head()

    merged_movie_df= pd.merge(movie_ids_ratings, movie_ids_titles, on='movieId')

    merged_movie_df.head()

    merged_movie_df.groupby('title').describe()

    merged_movie_df.groupby('title')['rating'].mean().head()

    merged_movie_df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

    merged_movie_df.groupby('title')['rating'].count().sort_values(ascending=False).head()

    movie_rating_mean_count= pd.DataFrame(columns=['rating_mean','rating_count'])

    movie_rating_mean_count["rating_mean"]=merged_movie_df.groupby('title')['rating'].mean()

    movie_rating_mean_count["rating_count"]=merged_movie_df.groupby('title')['rating'].count()

    movie_rating_mean_count.head()

    plt.figure(figsize=(10,8))
    sns.set_style("darkgrid")

    movie_rating_mean_count['rating_mean'].hist(bins=30, color='purple')

    plt.figure(figsize=(10,8))
    sns.set_style("darkgrid")

    movie_rating_mean_count["rating_count"].hist(bins=33, color="green")

    plt.figure(figsize=(10,8))
    sns.set_style("darkgrid")

    sns.regplot(x="rating_mean",y="rating_count",data=movie_rating_mean_count, color="brown")

    movie_rating_mean_count.sort_values("rating_count",ascending=False).head()

    user_movie_rating_matrix= merged_movie_df.pivot_table(index="userId",columns="title",values="rating")

    user_movie_rating_matrix

    user_movie_rating_matrix.shape

    pulp_fiction_ratings= user_movie_rating_matrix["Pulp Fiction (1994)"]

    pulp_fiction_correlations=pd.DataFrame(user_movie_rating_matrix.corrwith(pulp_fiction_ratings),columns=["pf_corr"])

    pulp_fiction_correlations.sort_values("pf_corr",ascending=False).head(5)
 
    pulp_fiction_correlations=pulp_fiction_correlations.join(movie_rating_mean_count["rating_count"])

    pulp_fiction_correlations.head()

    pulp_fiction_correlations.dropna(inplace=True)

    pulp_fiction_correlations.sort_values("pf_corr",ascending=False).head()

    pulp_fiction_correlations_50=pulp_fiction_correlations[pulp_fiction_correlations['rating_count']>50]

    pulp_fiction_correlations_50.sort_values("pf_corr",ascending=False).head()

    all_movie_correlations=user_movie_rating_matrix.corr(method="pearson",min_periods=50)

    all_movie_correlations.head()

    movie_data=[['Forrest Gump (1994)',4.0],['Fight Club (1999)',3.5],['Interstellar (2014)',4.0]]

    test_movies=pd.DataFrame(movie_data,columns=['Movie_Name','Movie_Rating'])

    test_movies.head()

    print(test_movies['Movie_Name'][0])
    print(test_movies['Movie_Rating'][0])

    recommended_movies_list = []

    for i in range(0,2):
        movie=all_movie_correlations[test_movies['Movie_Name'][i]].dropna()
        movie=movie.map(lambda movie_corr: movie_corr*test_movies["Movie_Rating"][i])
        recommended_movies_list.append(movie)

        recommended_movies = pd.concat(recommended_movies_list)

        recommended_movies

        recommended_movies.sort_values(inplace=True, ascending=False)

print("Movies related to Forest Gump:")

print(recommended_movies.head(10))

