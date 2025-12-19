import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from collaborative_filtering import CollaborativeFiltering
from nlp_matching import NLPMatching
from hybrid_engine import HybridRecommender
from data_preprocessing import DataPreprocessor

class TestScholarshipRecommender(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        print("Setting up test environment...")
        
        # Generate small test dataset
        self.test_students = pd.DataFrame({
            'student_id': [f'TEST_STU{i}' for i in range(1, 6)],
            'name': [f'Test Student {i}' for i in range(1, 6)],
            'major': ['Computer Science', 'Business', 'Engineering', 'Arts', 'Medicine'],
            'gpa': [3.8, 3.2, 3.9, 3.5, 3.7],
            'country': ['USA', 'India', 'UK', 'Canada', 'Australia'],
            'interests': ['AI, Web Development', 'Business Management', 'Engineering, Robotics', 
                         'Creative Arts, Design', 'Medical Research, Biology'],
            'background': ['CS undergraduate', 'Business student', 'Engineering major', 
                          'Arts student', 'Pre-med student'],
            'study_level': ['Undergraduate', 'Masters', 'PhD', 'Undergraduate', 'Masters']
        })
        
        self.test_scholarships = pd.DataFrame({
            'scholarship_id': [f'TEST_SCH{i}' for i in range(1, 11)],
            'title': [f'Test Scholarship {i}' for i in range(1, 11)],
            'field': ['Computer Science', 'Business', 'Engineering', 'Arts', 'Medicine',
                     'Computer Science', 'Business', 'Engineering', 'Arts', 'Medicine'],
            'eligibility': ['GPA 3.0+', 'GPA 3.2+', 'GPA 3.5+', 'GPA 3.0+', 'GPA 3.7+',
                           'GPA 3.0+', 'GPA 3.2+', 'GPA 3.5+', 'GPA 3.0+', 'GPA 3.7+'],
            'description': ['Scholarship for CS students', 'Business scholarship', 
                           'Engineering grant', 'Arts funding', 'Medical scholarship',
                           'Another CS scholarship', 'Another business scholarship',
                           'Another engineering grant', 'Another arts funding', 
                           'Another medical scholarship']
        })
        
        # Generate test ratings
        test_ratings = []
        for student_id in self.test_students['student_id']:
            for scholarship_id in self.test_scholarships['scholarship_id'][:3]:  # Rate first 3 scholarships
                rating = np.random.randint(3, 6)  # Ratings between 3-5
                test_ratings.append({
                    'student_id': student_id,
                    'scholarship_id': scholarship_id,
                    'rating': rating
                })
        
        self.test_ratings = pd.DataFrame(test_ratings)
        
        # Save test data
        os.makedirs('tests/test_data', exist_ok=True)
        self.test_students.to_csv('tests/test_data/test_students.csv', index=False)
        self.test_scholarships.to_csv('tests/test_data/test_scholarships.csv', index=False)
        self.test_ratings.to_csv('tests/test_data/test_ratings.csv', index=False)
        
        print("Test data generated successfully")
    
    def test_data_preprocessing(self):
        """Test data preprocessing pipeline"""
        print("\nTesting data preprocessing...")
        
        preprocessor = DataPreprocessor()
        
        # Load test data
        preprocessor.students_df = self.test_students
        preprocessor.scholarships_df = self.test_scholarships
        preprocessor.ratings_df = self.test_ratings
        
        # Run preprocessing
        preprocessor.preprocess_students()
        preprocessor.preprocess_scholarships()
        preprocessor.create_interaction_matrix()
        
        # Assertions
        self.assertIn('cleaned_profile', preprocessor.students_df.columns)
        self.assertIn('cleaned_description', preprocessor.scholarships_df.columns)
        self.assertIsNotNone(preprocessor.interaction_matrix)
        
        print("✓ Data preprocessing test passed")
    
    def test_collaborative_filtering(self):
        """Test collaborative filtering component"""
        print("\nTesting collaborative filtering...")
        
        cf = CollaborativeFiltering()
        cf.ratings_df = self.test_ratings
        
        # Create Surprise dataset
        import surprise
        reader = surprise.Reader(rating_scale=(1, 5))
        cf.data = surprise.Dataset.load_from_df(
            self.test_ratings[['student_id', 'scholarship_id', 'rating']], 
            reader
        )
        
        # Train model
        cf.train_model(algorithm='svd')
        
        # Test predictions
        test_student = 'TEST_STU1'
        test_scholarship = 'TEST_SCH1'
        
        prediction = cf.predict_rating(test_student, test_scholarship)
        self.assertIsInstance(prediction, (int, float))
        
        print("✓ Collaborative filtering test passed")
    
    def test_nlp_matching(self):
        """Test NLP matching component"""
        print("\nTesting NLP matching...")
        
        nlp = NLPMatching()
        nlp.students_df = self.test_students
        nlp.scholarships_df = self.test_scholarships
        nlp.load_models()
        nlp.generate_embeddings()
        
        # Test semantic similarity
        test_student = 'TEST_STU1'
        recommendations = nlp.calculate_semantic_similarity(test_student, top_n=3)
        
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertLessEqual(len(recommendations), 3)
        
        print("✓ NLP matching test passed")
    
    def test_hybrid_recommender(self):
        """Test hybrid recommender system"""
        print("\nTesting hybrid recommender...")
        
        hybrid = HybridRecommender(cf_weight=0.6, nlp_weight=0.4)
        
        # Set test data
        hybrid.students_df = self.test_students
        hybrid.scholarships_df = self.test_scholarships
        
        # Mock the component models
        class MockCF:
            def get_top_recommendations(self, student_id, scholarships_df, top_n):
                return scholarships_df.head(top_n).assign(cf_score=0.8)
        
        class MockNLP:
            def hybrid_nlp_recommendations(self, student_id, top_n):
                return pd.DataFrame({
                    'scholarship_id': hybrid.scholarships_df['scholarship_id'].head(top_n),
                    'nlp_score': 0.7
                })
        
        hybrid.cf_model = MockCF()
        hybrid.nlp_model = MockNLP()
        
        # Test hybrid recommendations
        test_student = 'TEST_STU1'
        recommendations = hybrid.get_hybrid_recommendations(test_student, top_n=5)
        
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertIn('hybrid_score', recommendations.columns)
        
        print("✓ Hybrid recommender test passed")
    
    def test_cold_start_handling(self):
        """Test cold start scenario handling"""
        print("\nTesting cold start handling...")
        
        hybrid = HybridRecommender()
        hybrid.students_df = self.test_students
        hybrid.scholarships_df = self.test_scholarships
        
        # Create a mock ratings dataframe with no interactions for a new student
        hybrid.cf_model.ratings_df = self.test_ratings
        
        # Mock components
        class MockCF:
            def get_top_recommendations(self, student_id, scholarships_df, top_n):
                # Simulate failure for new student
                if student_id == 'NEW_STUDENT':
                    raise ValueError("No data for new student")
                return scholarships_df.head(top_n)
        
        class MockNLP:
            def hybrid_nlp_recommendations(self, student_id, top_n):
                return hybrid.scholarships_df.head(top_n).assign(nlp_score=0.8)
        
        hybrid.cf_model = MockCF()
        hybrid.nlp_model = MockNLP()
        
        # Test with new student (cold start)
        recommendations = hybrid.get_hybrid_recommendations('NEW_STUDENT', top_n=5)
        
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertGreater(len(recommendations), 0)
        
        print("✓ Cold start handling test passed")

def run_performance_tests():
    """Run performance and scalability tests"""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    import time
    
    # Test with different dataset sizes
    dataset_sizes = [100, 500, 1000]
    
    for size in dataset_sizes:
        print(f"\nTesting with {size} students...")
        
        # Generate larger test dataset
        from data.sample_data_generator import generate_sample_data
        students, scholarships, ratings = generate_sample_data(
            num_students=size, 
            num_scholarships=size//5
        )
        
        # Test recommendation time
        hybrid = HybridRecommender()
        hybrid.students_df = students
        hybrid.scholarships_df = scholarships
        hybrid.cf_model.ratings_df = ratings
        
        start_time = time.time()
        
        # Test recommendation for first student
        try:
            recommendations = hybrid.get_hybrid_recommendations(
                students.iloc[0]['student_id'], 
                top_n=10
            )
            end_time = time.time()
            
            print(f"  Recommendations generated in {end_time - start_time:.2f} seconds")
            print(f"  Number of recommendations: {len(recommendations)}")
            
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)
    
    # Run performance tests
    run_performance_tests()
    
    print("\n" + "="*50)
    print("ALL TESTS COMPLETED")
    print("="*50)