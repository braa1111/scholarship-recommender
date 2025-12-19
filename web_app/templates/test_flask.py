from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Test data
    student = {
        'student_id': 'STU0001',
        'name': 'Test Student', 
        'major': 'Computer Science',
        'gpa': 3.8,
        'interests': 'AI, Machine Learning'
    }
    
    recommendations = [
        {
            'title': 'Computer Science Excellence Scholarship',
            'field': 'Computer Science', 
            'hybrid_score': 0.95,
            'explanation': 'Perfect match for your major and interests',
            'scholarship_id': 'S0001'
        },
        {
            'title': 'Data Science Innovation Grant',
            'field': 'Computer Science',
            'hybrid_score': 0.87, 
            'explanation': 'Strong match with your AI interests',
            'scholarship_id': 'S0002'
        },
        {
            'title': 'Tech Leadership Award',
            'field': 'Computer Science',
            'hybrid_score': 0.76,
            'explanation': 'Good potential match',
            'scholarship_id': 'S0003'
        }
    ]
    
    return render_template('recommendations.html', 
                         student=student, 
                         recommendations=recommendations, 
                         top_n=3)

if __name__ == '__main__':
    print("🌐 Starting TEST Flask App...")
    print("📚 Open: http://127.0.0.1:5000")
    print("⏹️  Press Ctrl+C to stop")
    app.run(debug=True, host='127.0.0.1', port=5000)