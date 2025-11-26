import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.user_movie_matrix = None
        self.movie_similarity = None
        
    def load_sample_data(self):
        """Create sample movie and ratings data"""
        # Sample movies with genres
        movies_data = {
            'movie_id': range(1, 21),
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Forrest Gump', 'Inception', 'The Matrix',
                'Goodfellas', 'Fight Club', 'The Lord of the Rings',
                'Star Wars', 'The Avengers', 'Jurassic Park', 'Titanic',
                'Avatar', 'The Lion King', 'Toy Story', 'Finding Nemo',
                'The Incredibles', 'Frozen'
            ],
            'genres': [
                'Drama', 'Crime Drama', 'Action Superhero', 'Crime Thriller',
                'Drama Romance', 'Sci-Fi Thriller', 'Sci-Fi Action',
                'Crime Drama', 'Drama Thriller', 'Fantasy Adventure',
                'Sci-Fi Adventure', 'Action Superhero', 'Adventure Sci-Fi',
                'Drama Romance', 'Sci-Fi Adventure', 'Animation Family',
                'Animation Family', 'Animation Family', 'Animation Action',
                'Animation Family'
            ]
        }
        
        # Sample ratings (user_id, movie_id, rating)
        np.random.seed(42)
        ratings_data = []
        for user in range(1, 51):  # 50 users
            num_ratings = np.random.randint(5, 15)
            movies = np.random.choice(range(1, 21), num_ratings, replace=False)
            for movie in movies:
                rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
                ratings_data.append([user, movie, rating])
        
        self.movies_df = pd.DataFrame(movies_data)
        self.ratings_df = pd.DataFrame(ratings_data, columns=['user_id', 'movie_id', 'rating'])
        
        print("Sample data loaded successfully!")
        print(f"Movies: {len(self.movies_df)}")
        print(f"Ratings: {len(self.ratings_df)}")
        print(f"Users: {self.ratings_df['user_id'].nunique()}")
    
    def content_based_filtering(self):
        """Build content-based recommendation using movie genres"""
        # Create TF-IDF matrix from genres
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies_df['genres'])
        
        # Calculate cosine similarity between movies
        self.movie_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("\nContent-based model trained!")
    
    def get_content_recommendations(self, movie_title, n=5):
        """Get recommendations based on movie content similarity"""
        if self.movie_similarity is None:
            print("Please train content-based model first!")
            return []
        
        # Find movie index
        idx = self.movies_df[self.movies_df['title'] == movie_title].index
        if len(idx) == 0:
            print(f"Movie '{movie_title}' not found!")
            return []
        
        idx = idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.movie_similarity[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]  # Exclude the movie itself
        
        movie_indices = [i[0] for i in sim_scores]
        recommendations = self.movies_df.iloc[movie_indices][['title', 'genres']].copy()
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        return recommendations
    
    def collaborative_filtering_user_based(self):
        """Build user-based collaborative filtering model"""
        # Create user-movie matrix
        self.user_movie_matrix = self.ratings_df.pivot_table(
            index='user_id',
            columns='movie_id',
            values='rating'
        ).fillna(0)
        
        print("\nUser-based collaborative filtering model ready!")
    
    def get_user_based_recommendations(self, user_id, n=5):
        """Get recommendations using user-based collaborative filtering"""
        if self.user_movie_matrix is None:
            print("Please train collaborative filtering model first!")
            return []
        
        if user_id not in self.user_movie_matrix.index:
            print(f"User {user_id} not found!")
            return []
        
        # Calculate user similarity
        user_similarity = cosine_similarity(self.user_movie_matrix)
        user_similarity_df = pd.DataFrame(
            user_similarity,
            index=self.user_movie_matrix.index,
            columns=self.user_movie_matrix.index
        )
        
        # Find similar users
        similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
        
        # Get movies rated by similar users but not by target user
        user_movies = set(self.ratings_df[
            (self.ratings_df['user_id'] == user_id) & 
            (self.ratings_df['rating'] >= 4)
        ]['movie_id'])
        
        recommendations = {}
        for sim_user, similarity in similar_users.items():
            sim_user_movies = self.ratings_df[
                (self.ratings_df['user_id'] == sim_user) & 
                (self.ratings_df['rating'] >= 4)
            ]['movie_id'].values
            
            for movie in sim_user_movies:
                if movie not in user_movies:
                    if movie not in recommendations:
                        recommendations[movie] = 0
                    recommendations[movie] += similarity
        
        # Sort and get top N
        top_movies = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
        
        result = []
        for movie_id, score in top_movies:
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
            result.append({
                'title': movie_info['title'].values[0],
                'genres': movie_info['genres'].values[0],
                'score': score
            })
        
        return pd.DataFrame(result)
    
    def collaborative_filtering_matrix_factorization(self, n_factors=10):
        """Build matrix factorization model using SVD"""
        if self.user_movie_matrix is None:
            self.collaborative_filtering_user_based()
        
        # Convert to sparse matrix
        matrix = csr_matrix(self.user_movie_matrix.values)
        
        # Perform SVD
        U, sigma, Vt = svds(matrix, k=n_factors)
        sigma = np.diag(sigma)
        
        # Reconstruct ratings
        self.predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        self.predicted_ratings_df = pd.DataFrame(
            self.predicted_ratings,
            columns=self.user_movie_matrix.columns,
            index=self.user_movie_matrix.index
        )
        
        print("\nMatrix factorization model trained!")
    
    def get_mf_recommendations(self, user_id, n=5):
        """Get recommendations using matrix factorization"""
        if self.predicted_ratings_df is None:
            print("Please train matrix factorization model first!")
            return []
        
        if user_id not in self.predicted_ratings_df.index:
            print(f"User {user_id} not found!")
            return []
        
        # Get user's predicted ratings
        user_predictions = self.predicted_ratings_df.loc[user_id]
        
        # Get movies user hasn't rated
        rated_movies = self.ratings_df[
            self.ratings_df['user_id'] == user_id
        ]['movie_id'].values
        
        # Filter out rated movies
        recommendations = user_predictions.drop(rated_movies, errors='ignore')
        recommendations = recommendations.sort_values(ascending=False)[:n]
        
        result = []
        for movie_id, pred_rating in recommendations.items():
            movie_info = self.movies_df[self.movies_df['movie_id'] == movie_id]
            result.append({
                'title': movie_info['title'].values[0],
                'genres': movie_info['genres'].values[0],
                'predicted_rating': pred_rating
            })
        
        return pd.DataFrame(result)


# Example usage
if __name__ == "__main__":
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Load sample data
    recommender.load_sample_data()
    
    print("\n" + "="*60)
    print("CONTENT-BASED FILTERING")
    print("="*60)
    
    # Train content-based model
    recommender.content_based_filtering()
    
    # Get content-based recommendations
    print("\nMovies similar to 'The Dark Knight':")
    recommendations = recommender.get_content_recommendations('The Dark Knight', n=5)
    print(recommendations.to_string(index=False))
    
    print("\n" + "="*60)
    print("USER-BASED COLLABORATIVE FILTERING")
    print("="*60)
    
    # Train collaborative filtering
    recommender.collaborative_filtering_user_based()
    
    # Get user-based recommendations
    print("\nRecommendations for User 1:")
    recommendations = recommender.get_user_based_recommendations(1, n=5)
    print(recommendations.to_string(index=False))
    
    print("\n" + "="*60)
    print("MATRIX FACTORIZATION (SVD)")
    print("="*60)
    
    # Train matrix factorization model
    recommender.collaborative_filtering_matrix_factorization(n_factors=5)
    
    # Get matrix factorization recommendations
    print("\nRecommendations for User 1 (Matrix Factorization):")
    recommendations = recommender.get_mf_recommendations(1, n=5)
    print(recommendations.to_string(index=False))
