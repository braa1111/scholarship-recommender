import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate
from surprise import KNNBasic
import pickle
import os

class CollaborativeFiltering:
    def __init__(self):
        self.model = None
        self.ratings_df = None
        self.trainset = None
        self.testset = None
        
    def load_data(self, ratings_path='data/processed/ratings_processed.csv'):
        """Load and prepare ratings data for Surprise library"""
        self.ratings_df = pd.read_csv(ratings_path)
        
        # Define rating scale
        reader = Reader(rating_scale=(1, 5))
        
        # Load data into Surprise dataset format
        self.data = Dataset.load_from_df(
            self.ratings_df[['student_id', 'scholarship_id', 'rating']], 
            reader
        )
        
        print(f"Loaded {len(self.ratings_df)} ratings for collaborative filtering")
    
    def train_model(self, algorithm='svd', test_size=0.2, random_state=42):
        """Train collaborative filtering model"""
        # Split data
        self.trainset, self.testset = train_test_split(self.data, test_size=test_size, random_state=random_state)
        
        # Choose algorithm
        if algorithm == 'svd':
            self.model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
        elif algorithm == 'knn':
            self.model = KNNBasic(k=40, sim_options={'name': 'cosine', 'user_based': True})
        else:
            raise ValueError("Algorithm must be 'svd' or 'knn'")
        
        # Train model
        self.model.fit(self.trainset)
        
        print(f"Collaborative Filtering model trained with {algorithm}")
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if self.model is None or self.testset is None:
            raise ValueError("Model must be trained first")
        
        # Make predictions on test set
        predictions = self.model.test(self.testset)
        
        # Calculate metrics
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        print(f"Model Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return rmse, mae
    
    def cross_validate(self, algorithm='svd', cv=5):
        """Perform cross-validation"""
        if algorithm == 'svd':
            model = SVD()
        elif algorithm == 'knn':
            model = KNNBasic()
        else:
            raise ValueError("Algorithm must be 'svd' or 'knn'")
        
        cv_results = cross_validate(model, self.data, measures=['RMSE', 'MAE'], cv=cv, verbose=True)
        
        return cv_results
    
    def predict_rating(self, student_id, scholarship_id):
        """Predict rating for a specific student-scholarship pair"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        try:
            prediction = self.model.predict(student_id, scholarship_id)
            return prediction.est
        except:
            return 0  # Return default rating if prediction fails
    
    def get_top_recommendations(self, student_id, scholarships_df, top_n=10):
        """Get top recommendations for a specific student"""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get all scholarship IDs
        all_scholarship_ids = scholarships_df['scholarship_id'].unique()
        
        # Predict ratings for all scholarships
        predictions = []
        for scholarship_id in all_scholarship_ids:
            predicted_rating = self.predict_rating(student_id, scholarship_id)
            predictions.append({
                'scholarship_id': scholarship_id,
                'predicted_rating': predicted_rating
            })
        
        # Convert to DataFrame and merge with scholarship info
        predictions_df = pd.DataFrame(predictions)
        recommendations_df = predictions_df.merge(
            scholarships_df, on='scholarship_id', how='left'
        )
        
        # Sort by predicted rating and return top N
        top_recommendations = recommendations_df.sort_values(
            'predicted_rating', ascending=False
        ).head(top_n)
        
        return top_recommendations
    
    def save_model(self, filepath='models/trained_models/cf_model.pkl'):
        """Save trained model to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/trained_models/cf_model.pkl'):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")

# Usage example
def train_collaborative_filtering():
    cf = CollaborativeFiltering()
    cf.load_data()
    cf.train_model(algorithm='svd')
    cf.evaluate_model()
    cf.save_model()
    return cf

if __name__ == "__main__":
    cf_model = train_collaborative_filtering()