from surprise import SVD, Dataset, Reader
import pandas as pd

def load_data():
    data = pd.read_csv('../data/ml-latest-small/ratings.csv')
    movies = pd.read_csv('../data/ml-latest-small/movies.csv')
    return data, movies

def train_model(data):
    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.25)
    svd = SVD()
    svd.fit(trainset)
    return svd

def get_recommendations(svd, data, user_id, n=10):
    user_ratings = data[data['userId'] == user_id]
    user_watched = user_ratings['movieId'].tolist()
    all_movies = data['movieId'].unique()
    recommendations = []

    for movie_id in all_movies:
        if movie_id not in user_watched:
            pred = svd.predict(user_id, movie_id)
            recommendations.append((movie_id, pred.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:n]

# Example usage:
if __name__ == "__main__":
    data, movies = load_data()
    svd = train_model(data)
    recommendations = get_recommendations(svd, data, 1, 10)
    recommended_movie_ids = [movie_id for movie_id, _ in recommendations]
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)]
    print(recommended_movies)
