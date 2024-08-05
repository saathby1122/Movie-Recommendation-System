def print_welcome_message():
    print("Welcome to the Movie Recommendation System!")

def format_recommendations(recommendations, movies):
    formatted_recs = []
    for movie_id, rating in recommendations:
        movie_title = movies[movies['movieId'] == movie_id]['title'].values[0]
        formatted_recs.append(f"{movie_title}: {rating:.2f}")
    return formatted_recs

if __name__ == "__main__":
    print_welcome_message()
