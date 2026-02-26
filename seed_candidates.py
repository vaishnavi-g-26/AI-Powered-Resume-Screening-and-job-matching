"""
Run this script ONCE to seed fake candidates and applications.
Command: python seed_candidates.py
"""

from app import app, db, User, Application, Job
from werkzeug.security import generate_password_hash
import random

# ===============================
# FAKE CANDIDATES
# ===============================

candidates = [
    # Indian candidates
    {"name": "Aarav Sharma",        "email": "aarav.sharma@gmail.com"},
    {"name": "Priya Patel",         "email": "priya.patel@gmail.com"},
    {"name": "Rohan Mehta",         "email": "rohan.mehta@gmail.com"},
    {"name": "Sneha Kulkarni",      "email": "sneha.kulkarni@gmail.com"},
    {"name": "Vikram Nair",         "email": "vikram.nair@gmail.com"},
    {"name": "Ananya Iyer",         "email": "ananya.iyer@gmail.com"},
    {"name": "Karan Desai",         "email": "karan.desai@gmail.com"},
    {"name": "Divya Reddy",         "email": "divya.reddy@gmail.com"},
    {"name": "Arjun Joshi",         "email": "arjun.joshi@gmail.com"},
    {"name": "Pooja Bhatt",         "email": "pooja.bhatt@gmail.com"},
    {"name": "Rahul Gupta",         "email": "rahul.gupta@gmail.com"},
    {"name": "Nisha Tiwari",        "email": "nisha.tiwari@gmail.com"},
    {"name": "Siddharth Rao",       "email": "siddharth.rao@gmail.com"},
    {"name": "Kavya Menon",         "email": "kavya.menon@gmail.com"},
    {"name": "Amit Chaudhary",      "email": "amit.chaudhary@gmail.com"},
    {"name": "Riya Singhania",      "email": "riya.singhania@gmail.com"},
    # International candidates
    {"name": "James Carter",        "email": "james.carter@gmail.com"},
    {"name": "Emily Zhang",         "email": "emily.zhang@gmail.com"},
    {"name": "Mohammed Al-Rashid",  "email": "mohammed.alrashid@gmail.com"},
    {"name": "Sofia Rossi",         "email": "sofia.rossi@gmail.com"},
    {"name": "Liam O'Brien",        "email": "liam.obrien@gmail.com"},
    {"name": "Yuki Tanaka",         "email": "yuki.tanaka@gmail.com"},
]

# Skills pool for generating fake resume matches
skill_pools = [
    "python flask django rest api sql git postgresql docker kubernetes aws",
    "python sql machine learning pandas statistics tensorflow scikit",
    "html css javascript react nodejs bootstrap frontend",
    "docker kubernetes aws linux ci/cd devops automation",
    "network security ethical hacking firewalls siem cybersecurity",
    "python django postgresql rest api backend microservices",
    "figma adobe xd wireframing prototyping ux design",
    "seo sem social media marketing google ads analytics",
    "agile scrum project management planning team leadership",
    "excel financial modeling budgeting analysis reporting",
    "recruitment payroll hr employee relations onboarding",
    "python flask sql git rest api backend developer",
    "html css react javascript frontend bootstrap responsive",
    "aws azure cloud architecture devops infrastructure",
    "postgresql mysql database backup performance tuning",
    "photoshop illustrator branding graphic design creativity",
    "content writing seo blogging analytics marketing strategy",
    "tally gst taxation accounting finance bookkeeping",
    "hiring screening interviewing hr tools talent acquisition",
    "requirement gathering documentation sql business analysis",
    "python tensorflow deep learning nlp computer vision",
    "java spring boot microservices sql maven git ci/cd",
]

def calculate_fake_score(resume_text, job_description):
    job_skills = [s.strip().lower() for s in job_description.split(",") if s.strip()]
    matched = [s for s in job_skills if s in resume_text.lower()]
    missing = [s for s in job_skills if s not in resume_text.lower()]
    score = round((len(matched) / len(job_skills)) * 100, 2) if job_skills else 0
    matched_str = ", ".join(matched)
    missing_str = ", ".join(missing)
    if score >= 75:
        rec = "Strong Fit ✅ — Excellent match for this role. Highly recommended."
    elif score >= 50:
        rec = "Moderate Fit 🟡 — Meets several requirements. Some skill gaps exist."
    else:
        rec = "Weak Fit ❌ — Limited skill match. Needs improvement in key areas."
    return score, matched_str, missing_str, rec

with app.app_context():

    created_count = 0
    skipped_count = 0

    for i, cand in enumerate(candidates):
        # Skip if email already exists
        existing = User.query.filter_by(email=cand["email"]).first()
        if existing:
            print(f"⚠ Skipped (already exists): {cand['name']}")
            skipped_count += 1
            continue

        # Create user
        user = User(
            name=cand["name"],
            email=cand["email"],
            password=generate_password_hash("password123"),
            role="candidate"
        )
        db.session.add(user)
        db.session.flush()  # Get user.id before commit

        # Pick a random resume skill set
        resume_text = skill_pools[i % len(skill_pools)]

        # Apply to ALL jobs
        jobs = Job.query.all()
        for job in jobs:
            score, matched, missing, rec = calculate_fake_score(resume_text, job.description)

            application = Application(
                user_id=user.id,
                job_id=job.id,
                resume_data=b"fake_resume_data",
                score=score,
                matched_skills=matched,
                missing_skills=missing,
                recommendation=rec
            )
            db.session.add(application)

        created_count += 1
        print(f"✅ Added: {cand['name']}")

    db.session.commit()
    print(f"\n🎉 Done! Created {created_count} candidates, skipped {skipped_count}.")
    print("Now go to any job's 'View Applications' page to see the ranked candidates!")