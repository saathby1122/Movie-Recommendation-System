import pandas as pd

def load_data():
    ratings = pd.read_csv('../data/ml-latest-small/ratings.csv')
    movies = pd.read_csv('../data/ml-latest-small/movies.csv')
    return ratings, movies

def preprocess_data(ratings):
    # Normalize ratings
    ratings['rating'] = ratings['rating'] / ratings['rating'].max()
    return ratings

if __name__ == "__main__":
    ratings, movies = load_data()
    ratings = preprocess_data(ratings)
    ratings.to_csv('../data/preprocessed_ratings.csv', index=False)
    movies.to_csv('../data/preprocessed_movies.csv', index=False)
