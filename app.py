from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
import io

from resume_parser import read_pdf, read_docx, clean_text
from matcher import calculate_match
from classifier import predict_job_category, train_classifier  # ✅ NEW: ML job classifier


# ===============================
# APP CONFIG
# ===============================

app = Flask(__name__)
app.secret_key = "supersecretkey"

app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:1457@localhost/resume_system"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ✅ Train the ML classifier once when app starts
train_classifier()


# ===============================
# DATABASE MODELS
# ===============================

class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))
    role = db.Column(db.String(20))


class Job(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    display_order = db.Column(db.Integer)


class Application(db.Model):
    __tablename__ = "applications"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    job_id = db.Column(db.Integer, db.ForeignKey("jobs.id"))
    resume_data = db.Column(db.LargeBinary)
    score = db.Column(db.Float)
    matched_skills = db.Column(db.Text)
    missing_skills = db.Column(db.Text)
    recommendation = db.Column(db.Text)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ===============================
# HOME
# ===============================

@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# REGISTER
# ===============================

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = generate_password_hash(request.form.get("password"))
        role = request.form.get("role")

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered")
            return redirect(url_for("register"))

        user = User(name=name, email=email, password=password, role=role)
        db.session.add(user)
        db.session.commit()
        login_user(user)

        if role == "admin":
            return redirect(url_for("admin_dashboard"))
        else:
            return redirect(url_for("candidate_dashboard"))

    return render_template("register.html")


# ===============================
# LOGIN
# ===============================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            if user.role == "admin":
                return redirect(url_for("admin_dashboard"))
            elif user.role == "candidate":
                return redirect(url_for("candidate_dashboard"))
            else:
                flash("Invalid role assigned.")
                return redirect(url_for("login"))
        else:
            flash("Invalid email or password")

    return render_template("login.html")


# ===============================
# LOGOUT
# ===============================

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("home"))


# ===============================
# ADMIN DASHBOARD
# ===============================

@app.route("/admin")
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        return redirect(url_for("home"))
    jobs = Job.query.all()
    return render_template("admin_dashboard.html", jobs=jobs)


@app.route("/add_job", methods=["POST"])
@login_required
def add_job():
    if current_user.role != "admin":
        return redirect(url_for("home"))
    title = request.form["title"]
    description = request.form["description"]
    job = Job(title=title, description=description)
    db.session.add(job)
    db.session.commit()
    return redirect(url_for("admin_dashboard"))


# ===============================
# SEED JOBS
# ===============================

@app.route("/seed_jobs")
def seed_jobs():
    jobs_list = [
        Job(title="Python Developer",          description="Python, Flask, Django, REST API, SQL, Git"),
        Job(title="Full Stack Developer",       description="HTML, CSS, JavaScript, React, Node.js, SQL"),
        Job(title="Digital Marketing Executive",description="SEO, SEM, Social Media Marketing, Google Ads"),
        Job(title="Data Scientist",             description="Python, Pandas, Machine Learning, SQL, Statistics"),
        Job(title="Machine Learning Engineer",  description="Python, Scikit-learn, TensorFlow, Deep Learning"),
        Job(title="DevOps Engineer",            description="Docker, Kubernetes, AWS, CI/CD, Linux"),
        Job(title="Cyber Security Analyst",     description="Network Security, Ethical Hacking, Firewalls, SIEM"),
        Job(title="Backend Developer",          description="Python, Django, PostgreSQL, REST API"),
        Job(title="Frontend Developer",         description="HTML, CSS, JavaScript, React, Bootstrap"),
        Job(title="Cloud Engineer",             description="AWS, Azure, Cloud Architecture, DevOps"),
        Job(title="Database Administrator",     description="PostgreSQL, MySQL, Backup, Performance Tuning"),
        Job(title="UI/UX Designer",             description="Figma, Adobe XD, Wireframing, Prototyping"),
        Job(title="Graphic Designer",           description="Photoshop, Illustrator, Branding, Creativity"),
        Job(title="Content Marketing Specialist",description="Content Writing, SEO, Blogging, Analytics"),
        Job(title="HR Executive",               description="Recruitment, Payroll, Employee Relations"),
        Job(title="Talent Acquisition Specialist",description="Hiring, Screening, Interviewing, HR Tools"),
        Job(title="Financial Analyst",          description="Excel, Financial Modeling, Budgeting"),
        Job(title="Accountant",                 description="Tally, GST, Taxation, Accounting"),
        Job(title="Project Manager",            description="Agile, Scrum, Team Management, Planning"),
        Job(title="Business Analyst",           description="Requirement Gathering, Documentation, SQL"),
    ]
    db.session.bulk_save_objects(jobs_list)
    db.session.commit()
    return "Jobs Added Successfully!"


# ===============================
# CANDIDATE DASHBOARD
# ===============================

@app.route('/candidate')
@login_required
def candidate_dashboard():
    jobs = Job.query.order_by(Job.display_order.asc()).all()
    return render_template('candidate_dashboard.html', jobs=jobs)


# ==========================
# JOB DETAILS PAGE
# ==========================

@app.route('/job/<int:job_id>')
@login_required
def job_details(job_id):
    job = Job.query.get_or_404(job_id)
    return render_template('job_details.html', job=job)


# ==========================
# APPLY ROUTE
# ==========================

@app.route("/apply/<int:job_id>", methods=["POST"])
@login_required
def apply(job_id):

    if current_user.role != "candidate":
        return redirect(url_for("home"))

    job = Job.query.get_or_404(job_id)
    file = request.files["resume"]

    if not file or file.filename == "":
        flash("Please upload a resume.")
        return redirect(url_for("job_details", job_id=job_id))

    # Step 1: Read raw binary (for storing in DB)
    resume_binary = file.read()

    # Step 2: Seek back so text extraction can read from start
    file.seek(0)

    # Step 3: Extract text based on file type
    if file.filename.endswith(".pdf"):
        resume_text = read_pdf(file)
    elif file.filename.endswith(".docx"):
        resume_text = read_docx(file)
    else:
        flash("Only PDF or DOCX allowed.")
        return redirect(url_for("job_details", job_id=job_id))

    # Step 4: Guard against empty resume text
    if not resume_text or resume_text.strip() == "":
        flash("Could not read resume content. Please try a different file.")
        return redirect(url_for("job_details", job_id=job_id))

    # Step 5: Clean text using NLTK (removes stopwords, lemmatizes, etc.)
    cleaned_resume = clean_text(resume_text)

    # Step 6: TF-IDF + Cosine Similarity score calculation
    result = calculate_match(cleaned_resume, job.description)

    if not result or len(result) != 4:
        flash("Matching failed. Please check your resume format.")
        return redirect(url_for("job_details", job_id=job_id))

    score, matched, missing, recommendation = result

    # ✅ Step 7: ML Job Category Prediction (Logistic Regression)
    # Uses the pre-trained classifier to predict which job category
    # this resume best matches, and returns top 3 with confidence %
    try:
        predicted_category, top3_matches = predict_job_category(cleaned_resume)
    except Exception:
        predicted_category = "Unable to predict"
        top3_matches = []

    # Step 8: Save application to PostgreSQL database
    application = Application(
        user_id=current_user.id,
        job_id=job.id,
        resume_data=resume_binary,
        score=float(score),           # convert numpy float64 → Python float
        matched_skills=matched,
        missing_skills=missing,
        recommendation=recommendation
    )
    db.session.add(application)
    db.session.commit()

    flash("Application submitted successfully!")

    # Step 9: Render result page with all AI analysis
    return render_template(
        "application_result.html",
        job=job,
        score=score,
        matched=matched,
        missing=missing,
        recommendation=recommendation,
        predicted_category=predicted_category,  # ✅ ML prediction
        top3_matches=top3_matches               # ✅ Top 3 job categories with confidence
    )


# ==========================
# VIEW APPLICATIONS (Admin)
# ==========================

@app.route("/admin/job/<int:job_id>/applications")
@login_required
def view_applications(job_id):
    if current_user.role != "admin":
        return redirect(url_for("home"))

    job = Job.query.get_or_404(job_id)
    applications = (
        db.session.query(Application, User)
        .join(User, Application.user_id == User.id)
        .filter(Application.job_id == job_id)
        .order_by(Application.score.desc())
        .all()
    )

    return render_template("view_applications.html", job=job, applications=applications)


# ==========================
# VIEW CANDIDATE DETAIL (Admin)
# ==========================

@app.route("/admin/application/<int:application_id>")
@login_required
def view_candidate_detail(application_id):
    if current_user.role != "admin":
        return redirect(url_for("home"))

    application = Application.query.get_or_404(application_id)
    candidate = User.query.get_or_404(application.user_id)
    job = Job.query.get_or_404(application.job_id)

    return render_template("candidate_detail.html",
                           application=application,
                           candidate=candidate,
                           job=job)


# ==========================
# DOWNLOAD RESUME (Admin)
# ==========================

@app.route("/admin/application/<int:application_id>/download")
@login_required
def download_resume(application_id):
    if current_user.role != "admin":
        return redirect(url_for("home"))

    application = Application.query.get_or_404(application_id)
    candidate = User.query.get_or_404(application.user_id)

    return send_file(
        io.BytesIO(application.resume_data),
        download_name=f"{candidate.name}_resume.pdf",
        as_attachment=True,
        mimetype="application/pdf"
    )


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    app.run(debug=True)