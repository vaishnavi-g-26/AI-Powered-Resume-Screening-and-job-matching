# test_system.py
import os
import pickle
from resume_parser import read_pdf, read_docx, clean_text
from classifier import calculate_match, predict_job_category

print("=" * 50)
print("🔍 TESTING RESUME SCREENING SYSTEM")
print("=" * 50)

# Test 1: Check if models exist
print("\n📊 Checking saved models:")
if os.path.exists('job_classifier.pkl'):
    print("✅ job_classifier.pkl found")
else:
    print("❌ job_classifier.pkl not found")

if os.path.exists('job_classifier_real.pkl'):
    print("✅ job_classifier_real.pkl found")
else:
    print("⚠️ job_classifier_real.pkl not found (optional)")

# Test 2: Test with sample resume text
print("\n📝 Testing with sample resume...")
sample_resume = """
Experienced Python Developer with 5 years of experience in Django, Flask, and Machine Learning.
Skilled in TensorFlow, Scikit-learn, and data analysis. Worked on multiple NLP projects.
Education: B.Tech in Computer Science from IIT.
"""

sample_job = """
Looking for a Machine Learning Engineer with Python, TensorFlow, and NLP experience.
Must have strong programming skills and data analysis background.
"""

# Test matching
score, matched, missing, recommendation = calculate_match(sample_resume, sample_job)
print(f"\n📊 Match Score: {score}%")
print(f"✅ Matched Skills: {matched}")
print(f"❌ Missing Skills: {missing}")
print(f"💡 Recommendation: {recommendation}")

# Test job category prediction
category, top3 = predict_job_category(sample_resume)
print(f"\n🎯 Predicted Job Category: {category}")
print("📊 Top 3 Matches:")
for job, prob in top3:
    print(f"   - {job}: {prob:.1f}%")

print("\n✅ Test complete!")