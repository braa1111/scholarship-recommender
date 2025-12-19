from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.secret_key = 'scholarship_recommender_secret_key_2025'

class SimpleRecommender:
    def __init__(self):
        self.scholarships_df = None
        
    def load_data(self):
        try:
            self.scholarships_df = pd.read_csv('../data/processed/scholarships_processed.csv')
            print("✅ Scholarships data loaded successfully")
        except:
            try:
                self.scholarships_df = pd.read_csv('../data/raw/scholarships.csv')
                print("✅ Raw scholarships data loaded successfully")
            except Exception as e:
                print(f"❌ Error loading data: {e}")
                self.create_sample_scholarships()
    
    def create_sample_scholarships(self):
        """Create sample scholarships data if files don't exist"""
        print("📝 Creating sample scholarships data...")
        
        self.scholarships_df = pd.DataFrame({
            'scholarship_id': ['S0001', 'S0002', 'S0003', 'S0004', 'S0005', 'S0006', 'S0007', 'S0008', 'S0009', 'S0010'],
            'title': ['Computer Science Excellence Scholarship', 'Business Leadership Award', 
                     'Engineering Innovation Grant', 'Arts Creativity Scholarship', 
                     'Medical Research Fellowship', 'Data Science Scholarship',
                     'Business Analytics Award', 'Civil Engineering Grant', 
                     'Visual Arts Scholarship', 'Medical Technology Fellowship'],
            'field': ['Computer Science', 'Business', 'Engineering', 'Arts', 'Medicine',
                     'Computer Science', 'Business', 'Engineering', 'Arts', 'Medicine'],
            'eligibility': ['GPA 3.0+, programming experience', 'GPA 3.2+, leadership qualities', 
                           'GPA 3.5+, engineering projects', 'GPA 3.0+, portfolio required', 
                           'GPA 3.7+, research experience', 'GPA 3.2+, Python experience',
                           'GPA 3.0+, analytics background', 'GPA 3.3+, civil projects',
                           'GPA 3.0+, art portfolio', 'GPA 3.5+, tech interest'],
            'description': [
                'Full scholarship for outstanding computer science students with programming experience',
                'Award for future business leaders with demonstrated leadership qualities',
                'Grant for innovative engineering projects and research initiatives',
                'Scholarship for creative arts students with strong portfolio submissions',
                'Fellowship for medical research and healthcare innovation projects',
                'Scholarship for data science students with Python and analytics skills',
                'Award for business analytics students with strong quantitative background',
                'Grant for civil engineering students with project experience',
                'Scholarship for visual arts students with creative portfolio',
                'Fellowship for medical technology and healthcare innovation'
            ],
            'deadline': ['2024-12-31', '2024-11-30', '2025-01-15', '2024-10-20', '2025-02-28',
                        '2024-12-15', '2025-03-31', '2025-01-20', '2024-11-15', '2025-04-30'],
            'funding_type': ['Full', 'Partial', 'Research Grant', 'Partial', 'Full',
                            'Full', 'Partial', 'Research Grant', 'Partial', 'Full'],
            'eligible_countries': ['USA, Canada, UK', 'All countries', 'USA, India, Germany', 
                                  'All countries', 'USA, UK, Australia', 'All countries', 
                                  'USA, Canada', 'All countries', 'USA, UK, Canada', 'All countries']
        })
        
        print("✅ Sample scholarships data created")
    
    def get_recommendations(self, student_data, top_n=5):
        try:
            student_name = student_data.get('name', 'Student')
            student_major = student_data.get('major', '')
            student_interests = student_data.get('interests', '')
            student_gpa = student_data.get('gpa', 0)
            
            recommendations = []
            for _, scholarship in self.scholarships_df.iterrows():
                score = 0
                
                if student_major.lower() in scholarship['field'].lower():
                    score += 0.6
                elif any(word in scholarship['field'].lower() for word in student_major.lower().split()):
                    score += 0.4
                
                if student_interests:
                    scholarship_text = (scholarship['title'] + ' ' + scholarship['description']).lower()
                    interest_words = [interest.strip().lower() for interest in student_interests.split(',')]
                    
                    for interest in interest_words:
                        if interest in scholarship_text:
                            score += 0.2
                            break
                
                try:
                    gpa = float(student_gpa)
                    if gpa > 3.5:
                        score += 0.1
                    elif gpa > 3.0:
                        score += 0.05
                except:
                    pass
                
                description_keywords = ['programming', 'coding', 'software', 'development', 'data', 'analysis', 
                                       'research', 'innovation', 'leadership', 'management', 'creative', 'design',
                                       'engineering', 'technology', 'medical', 'healthcare']
                
                scholarship_text = scholarship['description'].lower()
                for keyword in description_keywords:
                    if keyword in student_interests.lower() and keyword in scholarship_text:
                        score += 0.1
                        break
                
                score = min(score, 1.0)
                
                width_percentage = score * 100
                
                recommendations.append({
                    'scholarship_id': scholarship['scholarship_id'],
                    'title': scholarship['title'],
                    'field': scholarship['field'],
                    'hybrid_score': score,
                    'width_percentage': width_percentage,
                    'explanation': self.get_explanation(score, student_major, scholarship['field']),
                    'description': scholarship['description'],
                    'eligibility': scholarship['eligibility'],
                    'deadline': scholarship['deadline'],
                    'funding_type': scholarship['funding_type']
                })
            
            recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
            return recommendations[:top_n]
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return self.get_fallback_recommendations(top_n)
    
    def get_explanation(self, score, student_major, scholarship_field):
        if score >= 0.8:
            return f"Excellent match! Perfect alignment with your {student_major} background"
        elif score >= 0.6:
            return f"Strong match with your {student_major} major and interests"
        elif score >= 0.4:
            return f"Good potential match for {student_major} students"
        else:
            return f"Potential opportunity in {scholarship_field}"
    
    def get_fallback_recommendations(self, top_n=5):
        """Fallback recommendations if something goes wrong"""
        fallback_recs = [
            {
                'scholarship_id': 'S0001',
                'title': 'General Excellence Scholarship',
                'field': 'General',
                'hybrid_score': 0.8,
                'width_percentage': 80,
                'explanation': 'General scholarship opportunity for outstanding students',
                'description': 'Comprehensive scholarship for students with strong academic record',
                'eligibility': 'GPA 3.0+',
                'deadline': '2024-12-31',
                'funding_type': 'Partial'
            },
            {
                'scholarship_id': 'S0002', 
                'title': 'Academic Achievement Award',
                'field': 'General',
                'hybrid_score': 0.7,
                'width_percentage': 70,
                'explanation': 'Award for academic excellence and achievement',
                'description': 'Recognition award for students with exceptional academic performance',
                'eligibility': 'GPA 3.2+',
                'deadline': '2024-11-30',
                'funding_type': 'Partial'
            }
        ]
        return fallback_recs[:top_n]

recommender = SimpleRecommender()
recommender.load_data()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    student_name = request.form.get('student_name', '').strip()
    student_major = request.form.get('student_major', '').strip()
    student_interests = request.form.get('student_interests', '').strip()
    student_gpa = request.form.get('student_gpa', '').strip()
    top_n = int(request.form.get('top_n', 5))
    
    if not student_name or not student_major:
        return "Please fill in both name and major fields", 400
    
    print(f"🎯 Getting recommendations for {student_name} ({student_major})")
    
    student_data = {
        'name': student_name,
        'major': student_major,
        'interests': student_interests,
        'gpa': student_gpa
    }
    
    session['student_data'] = student_data
    
    recommendations = recommender.get_recommendations(student_data, top_n)
    
    student_info = {
        'name': student_name,
        'major': student_major,
        'interests': student_interests if student_interests else 'Not specified',
        'gpa': student_gpa if student_gpa else 'Not specified',
        'student_id': 'Custom Profile'
    }
    
    return render_template('recommendations.html', 
                         student=student_info, 
                         recommendations=recommendations,
                         top_n=top_n)

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """API endpoint for recommendations"""
    try:
        data = request.get_json()
        student_data = {
            'name': data.get('name', ''),
            'major': data.get('major', ''),
            'interests': data.get('interests', ''),
            'gpa': data.get('gpa', '')
        }
        top_n = data.get('top_n', 5)
        
        recommendations = recommender.get_recommendations(student_data, top_n)
        
        return {
            'student': student_data,
            'recommendations': recommendations
        }
        
    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    print("🌐 Starting Scholarship Recommender System...")
    print("📚 Open: http://127.0.0.1:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=10000)
