from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('job_dataset.csv').dropna(subset=['Job_Title'])

# Feature Extraction
tfidf = TfidfVectorizer(stop_words='english')
skills_matrix = tfidf.fit_transform(data['Job_Title'])
cosine_sim = cosine_similarity(skills_matrix, skills_matrix)

def recommend_jobs(age, height, top_n=5):
    # Filter data based on age and height
    filtered_data = data[
        (data['Age'] == age) & (data['Height'] == height)
    ]

    if filtered_data.empty:
        return []  # Return an empty list if no matches found

    job_indices = filtered_data.index.tolist()

    # Compute similarity scores for jobs based on existing jobs
    sim_scores = cosine_sim[job_indices].mean(axis=0)
    sorted_indices = sim_scores.argsort()[::-1]

    recommended_jobs = []
    seen_jobs = set()  # Track seen job titles to ensure uniqueness
    for idx in sorted_indices:
        job_title = data.iloc[idx]['Job_Title']
        if job_title not in seen_jobs:
            recommended_jobs.append(job_title)
            seen_jobs.add(job_title)  # Mark this job title as seen
            if len(recommended_jobs) >= top_n:
                break

    return recommended_jobs



@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        # Get age and height from the form
        age = int(request.form['age'])
        height = float(request.form['height'])
        
        print(f"Age: {age}, Height: {height}")  # Debugging line

        # Get job recommendations
        recommendations = recommend_jobs(age, height)
        
        print(f"Recommendations: {recommendations}")  # Debugging line

    return render_template('index.html', recommendations=recommendations)


if __name__ == "__main__":
    app.run(debug=True)

