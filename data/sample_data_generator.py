import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_sample_data(num_students=1000, num_scholarships=200):
    """Generate realistic sample data for testing"""
    
    # Fields and majors
    fields = ['Computer Science', 'Data Science', 'Engineering', 'Business', 
              'Arts', 'Medicine', 'Law', 'Education', 'Environmental Science']
    
    countries = ['USA', 'India', 'UK', 'Germany', 'Canada', 'Australia', 
                'UAE', 'Brazil', 'Japan', 'South Africa']
    
    interests_list = ['AI', 'Machine Learning', 'Web Development', 'Data Analysis',
                     'Business Management', 'Creative Arts', 'Medical Research',
                     'Environmental Conservation', 'Legal Studies', 'Education Technology']
    
    # Generate students dataset
    students = []
    for i in range(num_students):
        student = {
            'student_id': f'STU{i+1:04d}',
            'name': f'Student {i+1}',
            'major': random.choice(fields),
            'gpa': round(np.random.normal(3.2, 0.5), 2),
            'country': random.choice(countries),
            'interests': ', '.join(random.sample(interests_list, 3)),
            'background': f"Background description for student {i+1}",
            'study_level': random.choice(['Undergraduate', 'Masters', 'PhD'])
        }
        students.append(student)
    
    students_df = pd.DataFrame(students)
    
    # Generate scholarships dataset
    scholarships = []
    for i in range(num_scholarships):
        deadline = datetime.now() + timedelta(days=random.randint(30, 365))
        scholarship = {
            'scholarship_id': f'S{i+1:04d}',
            'title': f'{random.choice(fields)} Excellence Scholarship {i+1}',
            'field': random.choice(fields),
            'eligibility': f"Minimum GPA {random.choice([3.0, 3.2, 3.5])}, {random.choice(['International students welcome', 'Domestic students only'])}",
            'description': f"Comprehensive scholarship for {random.choice(fields)} students focusing on {random.choice(interests_list)}. {random.choice(['Full tuition coverage', 'Partial funding', 'Research grant'])} available.",
            'deadline': deadline.strftime('%Y-%m-%d'),
            'funding_type': random.choice(['Full', 'Partial', 'Research Grant']),
            'eligible_countries': ', '.join(random.sample(countries, 3))
        }
        scholarships.append(scholarship)
    
    scholarships_df = pd.DataFrame(scholarships)
    
    # Generate ratings/interactions dataset
    ratings = []
    for student in students[:800]:  # Only 80% of students have interactions (sparsity)
        num_ratings = random.randint(5, 20)
        rated_scholarships = random.sample(scholarships_df['scholarship_id'].tolist(), num_ratings)
        
        for scholarship_id in rated_scholarships:
            # Higher probability of good rating if field matches
            scholarship_field = scholarships_df[scholarships_df['scholarship_id'] == scholarship_id]['field'].iloc[0]
            base_rating = 4 if scholarship_field == student['major'] else 3
            rating = max(1, min(5, int(np.random.normal(base_rating, 0.8))))
            
            rating_data = {
                'student_id': student['student_id'],
                'scholarship_id': scholarship_id,
                'rating': rating,
                'interaction_type': random.choice(['click', 'save', 'application'])
            }
            ratings.append(rating_data)
    
    ratings_df = pd.DataFrame(ratings)
    
    # Save datasets
    students_df.to_csv('data/raw/students.csv', index=False)
    scholarships_df.to_csv('data/raw/scholarships.csv', index=False)
    ratings_df.to_csv('data/raw/ratings.csv', index=False)
    
    print(f"Generated {len(students_df)} students, {len(scholarships_df)} scholarships, {len(ratings_df)} ratings")
    return students_df, scholarships_df, ratings_df

if __name__ == "__main__":
    generate_sample_data()