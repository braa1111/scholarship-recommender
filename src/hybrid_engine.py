import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

from sympy import python

class HybridRecommender:
    def __init__(self, cf_weight=0.6, nlp_weight=0.4):
        self.cf_weight = cf_weight
        self.nlp_weight = nlp_weight
        self.scaler = MinMaxScaler()
        self.students_df = None
        self.scholarships_df = None
        
    def load_components(self):
        """Load both CF and NLP components"""
        print("🔄 Loading hybrid recommender components...")
        
        # Load data
        self.students_df = pd.read_csv('data/processed/students_processed.csv')
        self.scholarships_df = pd.read_csv('data/processed/scholarships_processed.csv')
        
        print("✅ Hybrid recommender components loaded successfully")
        print(f"📊 Students: {len(self.students_df)}, Scholarships: {len(self.scholarships_df)}")
    
    def get_simple_cf_recommendations(self, student_id, top_n=10):
        """Simple collaborative filtering based on field matching"""
        try:
            student = self.students_df[self.students_df['student_id'] == student_id].iloc[0]
            
            # Get students with same major
            similar_students = self.students_df[
                (self.students_df['major'] == student['major']) & 
                (self.students_df['student_id'] != student_id)
            ]
            
            if len(similar_students) > 0:
                # Simple CF: recommend scholarships that similar students might like
                cf_scores = []
                for _, scholarship in self.scholarships_df.iterrows():
                    score = 1.0 if scholarship['field'] == student['major'] else 0.3
                    cf_scores.append({
                        'scholarship_id': scholarship['scholarship_id'],
                        'cf_score': score,
                        'title': scholarship['title'],
                        'field': scholarship['field']
                    })
                
                return pd.DataFrame(cf_scores)
            else:
                # Fallback: field-based recommendations
                return self.get_field_based_recommendations(student_id, top_n)
                
        except Exception as e:
            print(f"CF recommendation failed: {e}")
            return self.get_field_based_recommendations(student_id, top_n)
    
    def get_nlp_recommendations(self, student_id, top_n=10):
        """Simple NLP-based recommendations using text matching"""
        try:
            student = self.students_df[self.students_df['student_id'] == student_id].iloc[0]
            
            nlp_scores = []
            for _, scholarship in self.scholarships_df.iterrows():
                # Simple text matching
                score = 0
                
                # Field match
                if student['major'] == scholarship['field']:
                    score += 0.6
                
                # Interest matching
                student_interests = student['interests'].lower().split(', ')
                scholarship_text = (scholarship['title'] + ' ' + scholarship['description']).lower()
                
                for interest in student_interests:
                    if interest in scholarship_text:
                        score += 0.2
                
                # GPA consideration
                if 'gpa' in student and student['gpa'] > 3.0:
                    score += 0.1
                if 'gpa' in student and student['gpa'] > 3.5:
                    score += 0.1
                
                nlp_scores.append({
                    'scholarship_id': scholarship['scholarship_id'],
                    'nlp_score': min(score, 1.0),  # Cap at 1.0
                    'title': scholarship['title'],
                    'field': scholarship['field']
                })
            
            return pd.DataFrame(nlp_scores)
            
        except Exception as e:
            print(f"NLP recommendation failed: {e}")
            return pd.DataFrame()
    
    def get_field_based_recommendations(self, student_id, top_n=10):
        """Fallback field-based recommendations"""
        student = self.students_df[self.students_df['student_id'] == student_id].iloc[0]
        
        field_scores = []
        for _, scholarship in self.scholarships_df.iterrows():
            score = 3.0 if scholarship['field'] == student['major'] else 1.0
            field_scores.append({
                'scholarship_id': scholarship['scholarship_id'],
                'cf_score': score,
                'title': scholarship['title'],
                'field': scholarship['field']
            })
        
        return pd.DataFrame(field_scores)
    
    def get_hybrid_recommendations(self, student_id, top_n=10, cold_start_handling=True):
        """Get hybrid recommendations combining CF and NLP approaches"""
        print(f"🎯 Getting hybrid recommendations for {student_id}...")
        
        # Load components if not already loaded
        if self.students_df is None:
            self.load_components()
        
        # Get CF recommendations
        cf_recommendations = self.get_simple_cf_recommendations(student_id, top_n * 2)
        
        # Get NLP recommendations
        nlp_recommendations = self.get_nlp_recommendations(student_id, top_n * 2)
        
        # Merge recommendations
        if not cf_recommendations.empty and not nlp_recommendations.empty:
            # Full hybrid approach
            all_recommendations = pd.merge(
                cf_recommendations[['scholarship_id', 'cf_score', 'title', 'field']],
                nlp_recommendations[['scholarship_id', 'nlp_score']],
                on='scholarship_id', 
                how='outer'
            )
        elif not cf_recommendations.empty:
            # CF-only fallback
            all_recommendations = cf_recommendations[['scholarship_id', 'cf_score', 'title', 'field']]
            all_recommendations['nlp_score'] = 0
        elif not nlp_recommendations.empty:
            # NLP-only fallback
            all_recommendations = nlp_recommendations[['scholarship_id', 'nlp_score', 'title', 'field']]
            all_recommendations['cf_score'] = 0
        else:
            # Ultimate fallback - field-based
            return self.get_field_based_recommendations(student_id, top_n)
        
        # Fill NaN values
        all_recommendations = all_recommendations.fillna(0)
        
        # Normalize scores
        if all_recommendations['cf_score'].max() > 0:
            all_recommendations['cf_normalized'] = (
                all_recommendations['cf_score'] / all_recommendations['cf_score'].max()
            )
        else:
            all_recommendations['cf_normalized'] = 0
            
        if all_recommendations['nlp_score'].max() > 0:
            all_recommendations['nlp_normalized'] = (
                all_recommendations['nlp_score'] / all_recommendations['nlp_score'].max()
            )
        else:
            all_recommendations['nlp_normalized'] = 0
        
        # Calculate hybrid score
        all_recommendations['hybrid_score'] = (
            self.cf_weight * all_recommendations['cf_normalized'] +
            self.nlp_weight * all_recommendations['nlp_normalized']
        )
        
        # Add explanation
        all_recommendations['explanation'] = all_recommendations.apply(
            self.generate_explanation, 
            axis=1
        )
        
        # Sort and return top N
        final_recommendations = all_recommendations.sort_values(
            'hybrid_score', ascending=False
        ).head(top_n)
        
        print(f"✅ Generated {len(final_recommendations)} hybrid recommendations")
        return final_recommendations
    
    def generate_explanation(self, row):
        """Generate explanation for why a scholarship was recommended"""
        explanations = []
        
        if row['cf_normalized'] > 0.7:
            explanations.append("Popular among students in your field")
        
        if row['nlp_score'] > 0.7:
            explanations.append("Strong match with your academic profile")
        
        if row['cf_normalized'] > 0.5 and row['nlp_normalized'] > 0.5:
            explanations.append("Excellent overall match")
        
        if not explanations:
            field = row['field'] if 'field' in row else 'this field'
            explanations.append(f"Good match for {field} students")
        
        return ". ".join(explanations)
    
    def get_popular_scholarships(self, top_n=10):
        """Get popular scholarships as fallback"""
        # Simple popularity based on field distribution
        field_counts = self.scholarships_df['field'].value_counts()
        popular_scholarships = self.scholarships_df.copy()
        popular_scholarships['popularity_score'] = popular_scholarships['field'].map(
            lambda x: field_counts.get(x, 0) / field_counts.max()
        )
        
        popular_scholarships['explanation'] = "Popular scholarship in this field"
        
        return popular_scholarships.sort_values('popularity_score', ascending=False).head(top_n)

def setup_hybrid_recommender():
    """Setup and return hybrid recommender"""
    hybrid_rec = HybridRecommender(cf_weight=0.6, nlp_weight=0.4)
    hybrid_rec.load_components()
    return hybrid_rec

# Test function
def test_hybrid_system():
    """Test the hybrid system"""
    print("🧪 Testing Hybrid Recommender System...")
    
    try:
        hybrid_model = setup_hybrid_recommender()
        recommendations = hybrid_model.get_hybrid_recommendations('STU0001', top_n=5)
        
        print('✅ Hybrid system working!')
        print('Sample recommendations for STU0001:')
        for i, row in recommendations.iterrows():
            print(f'{i+1}. {row["title"]} (Score: {row["hybrid_score"]:.3f}) - {row["explanation"]}')
            
        return True
        
    except Exception as e:
        print(f'❌ Hybrid system test failed: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_hybrid_system()