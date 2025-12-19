import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

class DataPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.scaler = MinMaxScaler()
    
    def load_data(self):
        """Load and merge all datasets"""
        self.students_df = pd.read_csv('data/raw/students.csv')
        self.scholarships_df = pd.read_csv('data/raw/scholarships.csv')
        self.ratings_df = pd.read_csv('data/raw/ratings.csv')
        
        print(f"Loaded {len(self.students_df)} students, {len(self.scholarships_df)} scholarships, {len(self.ratings_df)} ratings")
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords and lemmatize
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def preprocess_students(self):
        """Preprocess student data"""
        # Create combined text profile
        self.students_df['profile_text'] = (
            self.students_df['major'] + " " + 
            self.students_df['interests'] + " " + 
            self.students_df['background'] + " " +
            self.students_df['study_level']
        )
        
        # Clean profile text
        self.students_df['cleaned_profile'] = self.students_df['profile_text'].apply(self.clean_text)
        
        # Normalize GPA
        self.students_df['gpa_normalized'] = self.scaler.fit_transform(
            self.students_df[['gpa']]
        )
        
        print("Student data preprocessing completed")
    
    def preprocess_scholarships(self):
        """Preprocess scholarship data"""
        # Create combined text description
        self.scholarships_df['description_text'] = (
            self.scholarships_df['title'] + " " + 
            self.scholarships_df['field'] + " " + 
            self.scholarships_df['eligibility'] + " " + 
            self.scholarships_df['description']
        )
        
        # Clean description text
        self.scholarships_df['cleaned_description'] = self.scholarships_df['description_text'].apply(self.clean_text)
        
        print("Scholarship data preprocessing completed")
    
    def create_interaction_matrix(self):
        """Create user-item interaction matrix"""
        # Create pivot table for collaborative filtering
        self.interaction_matrix = self.ratings_df.pivot_table(
            index='student_id', 
            columns='scholarship_id', 
            values='rating', 
            fill_value=0
        )
        
        print(f"Interaction matrix shape: {self.interaction_matrix.shape}")
    
    def save_processed_data(self):
        """Save processed datasets"""
        self.students_df.to_csv('data/processed/students_processed.csv', index=False)
        self.scholarships_df.to_csv('data/processed/scholarships_processed.csv', index=False)
        self.ratings_df.to_csv('data/processed/ratings_processed.csv', index=False)
        
        print("Processed data saved successfully")
    
    def run_full_preprocessing(self):
        """Execute complete preprocessing pipeline"""
        self.load_data()
        self.preprocess_students()
        self.preprocess_scholarships()
        self.create_interaction_matrix()
        self.save_processed_data()
        
        return self.students_df, self.scholarships_df, self.ratings_df

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    students, scholarships, ratings = preprocessor.run_full_preprocessing()