# Create index.html
index_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Scholarship Recommender System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1>🎓 Scholarship Recommender System</h1>
        <p>Select a student to get personalized scholarship recommendations:</p>
        
        <form method="POST" action="/recommend">
            <select name="student_id" class="form-select">
                <option value="STU0001">STU0001 - Computer Science</option>
                <option value="STU0002">STU0002 - Business</option>
                <option value="STU0003">STU0003 - Engineering</option>
            </select>
            <button type="submit" class="btn btn-primary mt-3">Get Recommendations</button>
        </form>
    </div>
</body>
</html>'''

# Create recommendations.html
rec_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Recommendations</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>Recommendations for {{ student.name }}</h1>
        <a href="/" class="btn btn-secondary">← Back</a>
        
        {% for rec in recommendations %}
        <div class="card mt-3">
            <div class="card-body">
                <h5>{{ rec.title }}</h5>
                <p>Score: {{ "%.1f"|format(rec.hybrid_score * 100) }}% - {{ rec.explanation }}</p>
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>'''

# Write files
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(index_content)

with open('templates/recommendations.html', 'w', encoding='utf-8') as f:
    f.write(rec_content)

print("✅ Templates created successfully!")