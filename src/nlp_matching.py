import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import torch
import pickle
import os

class NLPMatching:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.sentence_model = None
        self.spacy_model = None
        self.student_embeddings = None
        self.scholarship_embeddings = None
        self.students_df = None
        self.scholarships_df = None
        
    def load_models(self):
        """Load NLP models"""
        self.sentence_model = SentenceTransformer(self.model_name)
        try:
            self.spacy_model = spacy.load('en_core_web_md')
        except OSError:
            print("spaCy model not found. Please download: python -m spacy download en_core_web_md")
            self.spacy_model = None
            
        print("NLP models loaded successfully")
    
    def load_data(self):
        """Load processed student and scholarship data"""
        self.students_df = pd.read_csv('data/processed/students_processed.csv')
        self.scholarships_df = pd.read_csv('data/processed/scholarships_processed.csv')
        
        print(f"Loaded {len(self.students_df)} students and {len(self.scholarships_df)} scholarships")
    
    def generate_embeddings(self):
        """Generate embeddings for student profiles and scholarship descriptions"""
        if self.sentence_model is None:
            self.load_models()
        
        # Generate student profile embeddings
        student_texts = self.students_df['cleaned_profile'].fillna('').tolist()
        self.student_embeddings = self.sentence_model.encode(student_texts, show_progress_bar=True)
        
        # Generate scholarship description embeddings
        scholarship_texts = self.scholarships_df['cleaned_description'].fillna('').tolist()
        self.scholarship_embeddings = self.sentence_model.encode(scholarship_texts, show_progress_bar=True)
        
        print("Embeddings generated successfully")
    
    def calculate_semantic_similarity(self, student_id, top_n=10):
        """Calculate semantic similarity between a student and all scholarships"""
        if self.student_embeddings is None or self.scholarship_embeddings is None:
            self.generate_embeddings()
        
        # Find student index
        student_idx = self.students_df[self.students_df['student_id'] == student_id].index[0]
        
        # Calculate cosine similarity
        student_embedding = self.student_embeddings[student_idx].reshape(1, -1)
        similarities = cosine_similarity(student_embedding, self.scholarship_embeddings)[0]
        
        # Create results dataframe
        results = []
        for i, similarity in enumerate(similarities):
            results.append({
                'scholarship_id': self.scholarships_df.iloc[i]['scholarship_id'],
                'semantic_similarity': similarity,
                'title': self.scholarships_df.iloc[i]['title'],
                'field': self.scholarships_df.iloc[i]['field']
            })
        
        results_df = pd.DataFrame(results)
        
        # Sort by similarity and return top N
        top_recommendations = results_df.sort_values('semantic_similarity', ascending=False).head(top_n)
        
        return top_recommendations
    
    def get_field_based_recommendations(self, student_id, top_n=10):
        """Get recommendations based on field matching"""
        student_data = self.students_df[self.students_df['student_id'] == student_id].iloc[0]
        student_major = student_data['major']
        student_interests = student_data['interests'].split(', ')
        
        # Calculate field matching score
        def calculate_field_score(scholarship_field, student_major, student_interests):
            score = 0
            # Exact field match
            if scholarship_field == student_major:
                score += 3
            # Partial match or interest match
            for interest in student_interests:
                if interest.lower() in scholarship_field.lower():
                    score += 1
            return score
        
        recommendations = []
        for _, scholarship in self.scholarships_df.iterrows():
            field_score = calculate_field_score(scholarship['field'], student_major, student_interests)
            
            recommendations.append({
                'scholarship_id': scholarship['scholarship_id'],
                'field_score': field_score,
                'title': scholarship['title'],
                'field': scholarship['field'],
                'match_reason': f"Field: {scholarship['field']} matches your major: {student_major}"
            })
        
        recommendations_df = pd.DataFrame(recommendations)
        top_recommendations = recommendations_df.sort_values('field_score', ascending=False).head(top_n)
        
        return top_recommendations
    
    def hybrid_nlp_recommendations(self, student_id, top_n=10, semantic_weight=0.7, field_weight=0.3):
        """Combine semantic similarity and field-based matching"""
        # Get semantic recommendations
        semantic_recs = self.calculate_semantic_similarity(student_id, top_n * 2)
        semantic_recs = semantic_recs.rename(columns={'semantic_similarity': 'semantic_score'})
        
        # Get field-based recommendations
        field_recs = self.get_field_based_recommendations(student_id, top_n * 2)
        
        # Merge and calculate hybrid score
        merged_recs = pd.merge(semantic_recs, field_recs, on='scholarship_id', how='outer')
        merged_recs = merged_recs.fillna(0)
        
        # Normalize scores
        merged_recs['semantic_normalized'] = merged_recs['semantic_score'] / merged_recs['semantic_score'].max()
        merged_recs['field_normalized'] = merged_recs['field_score'] / merged_recs['field_score'].max() if merged_recs['field_score'].max() > 0 else 0
        
        # Calculate hybrid score
        merged_recs['hybrid_score'] = (
            semantic_weight * merged_recs['semantic_normalized'] + 
            field_weight * merged_recs['field_normalized']
        )
        
        # Sort by hybrid score and return top N
        final_recommendations = merged_recs.sort_values('hybrid_score', ascending=False).head(top_n)
        
        return final_recommendations
    
    def save_embeddings(self):
        """Save generated embeddings for future use"""
        os.makedirs('models/trained_models', exist_ok=True)
        
        embeddings_data = {
            'student_embeddings': self.student_embeddings,
            'scholarship_embeddings': self.scholarship_embeddings,
            'student_ids': self.students_df['student_id'].tolist(),
            'scholarship_ids': self.scholarships_df['scholarship_id'].tolist()
        }
        
        with open('models/trained_models/nlp_embeddings.pkl', 'wb') as f:
            pickle.dump(embeddings_data, f)
        
        print("Embeddings saved successfully")
    
    def load_embeddings(self):
        """Load pre-generated embeddings"""
        with open('models/trained_models/nlp_embeddings.pkl', 'rb') as f:
            embeddings_data = pickle.load(f)
        
        self.student_embeddings = embeddings_data['student_embeddings']
        self.scholarship_embeddings = embeddings_data['scholarship_embeddings']
        
        print("Embeddings loaded successfully")

# Usage example
def setup_nlp_matching():
    nlp_matcher = NLPMatching()
    nlp_matcher.load_models()
    nlp_matcher.load_data()
    nlp_matcher.generate_embeddings()
    nlp_matcher.save_embeddings()
    return nlp_matcher

if __name__ == "__main__":
    nlp_model = setup_nlp_matching()
    
    # Test with a sample student
    sample_recommendations = nlp_model.hybrid_nlp_recommendations('STU0001', top_n=5)
    print("Sample NLP Recommendations:")
    print(sample_recommendations[['scholarship_id', 'title', 'hybrid_score']])