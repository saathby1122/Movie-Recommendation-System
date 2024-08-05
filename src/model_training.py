from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pandas as pd

def load_data():
    data = pd.read_csv('../data/preprocessed_ratings.csv')
    reader = Reader(rating_scale=(0.5, 1.0))
    dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    return dataset

def train_model(dataset):
    trainset, testset = train_test_split(dataset, test_size=0.25)
    svd = SVD()
    svd.fit(trainset)
    return svd, trainset, testset

if __name__ == "__main__":
    dataset = load_data()
    svd, trainset, testset = train_model(dataset)
    # Save model if needed (requires additional libraries like pickle or joblib)
