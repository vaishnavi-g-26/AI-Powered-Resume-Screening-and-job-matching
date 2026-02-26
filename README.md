🚀 AI-Powered Resume Screening & Candidate Ranking System

🔍 Intelligent Hiring Using AI, ML & NLP

An AI-driven full-stack web application that automates resume screening, skill gap analysis, and candidate ranking using Machine Learning and Natural Language Processing. <br><br>


📌 Problem Statement

Companies receive hundreds of resumes per job opening. Manual screening is:

• Slow (20–40 resumes/hour)

• Inconsistent

• Prone to bias

• Expensive

This system automates the entire screening process using AI. <br><br>


💡 Solution

The system:

• Extracts resume text (PDF/DOCX)

• Cleans and processes text using NLP

• Identifies technical skills using spaCy

• Expands skills using semantic taxonomy

• Calculates match score using TF-IDF + Cosine Similarity

• Predicts job category using Logistic Regression (trained on 9,544 resumes)

• Ranks candidates automatically <br><br>


🧠 Tech Stack

• Python

• Flask

• PostgreSQL

• NLTK

• spaCy

• Scikit-learn

• pdfminer.six

• SQLAlchemy

• Flask-Login <br><br>


⚙️ AI Pipeline

Resume Upload → Text Extraction → NLP Cleaning →
Skill Recognition → Semantic Expansion →
TF-IDF Vectorization → Cosine Similarity →
ML Job Prediction → Ranked Output <br><br>

🏆 Key Features
For Candidates

• Instant AI match score

• Matched & Missing Skills

• Career Fit Prediction

• Skill Gap Report <br><br>


For HR / Admin

• Auto-ranked candidate dashboard

• Gold, Silver, Bronze ranking

• Download resume from database

• Data-driven hiring decisions <br><br>


📊 Scoring Formula

If all skills match → 100%

Otherwise:
Final Score = (Skill Match × 70%) + (TF-IDF Score × 30%) <br><br>


📷 Screenshots
![Home Page](screenshots/home_page.png)

![Admin Dashboard](screenshots/admin_dashboard.png)

![Application](screenshots/application.png)

![Candidate Dashboard](screenshots/candidate_dashboard.png)

![Apply Now](screenshots/apply_now.png)

![Match Score](screenshots/match_score.png)

<br><br>
📈 Business Impact

• Reduces screening time from hours to seconds

• Eliminates bias

• Provides structured candidate feedback

• Improves hiring efficiency <br><br>


👩‍💻 Developed By

Vaishnavi Gade
BSc Student | AI & Machine Learning Enthusiast
