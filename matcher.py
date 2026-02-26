"""
matcher.py
==========
Semantic AI Matching with Skill Taxonomy

Key feature: SKILL HIERARCHY
If a resume has "Machine Learning", it implicitly covers:
→ Scikit-learn, TensorFlow, PyTorch, Deep Learning, etc.

If a resume has "Python", it implicitly covers:
→ Flask, Django, Scripting, Automation, etc.

This mirrors real-world hiring — a Machine Learning expert
is expected to know TensorFlow even if not explicitly listed.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


# ================================================================
# SKILL TAXONOMY — Parent skill → implied child skills
# If resume has the PARENT, child skills are considered matched
# ================================================================

SKILL_TAXONOMY = {

    # ── Machine Learning / AI ────────────────────────────────────
    "machine learning": [
        "scikit-learn", "scikit learn", "sklearn",
        "tensorflow", "keras", "pytorch", "deep learning",
        "neural network", "neural networks", "gradient boosting",
        "xgboost", "lightgbm", "random forest", "decision tree",
        "regression", "classification", "clustering", "nlp",
        "natural language processing", "computer vision",
        "model training", "model deployment", "ml pipeline",
        "feature engineering", "data preprocessing",
    ],

    "deep learning": [
        "tensorflow", "keras", "pytorch", "neural network",
        "cnn", "rnn", "lstm", "transformer", "bert", "gpt",
        "computer vision", "nlp", "image recognition",
    ],

    "data science": [
        "machine learning", "pandas", "numpy", "matplotlib",
        "seaborn", "statistics", "data analysis", "data visualization",
        "scikit-learn", "jupyter", "r", "tableau", "power bi",
        "sql", "data wrangling", "feature engineering",
    ],

    "artificial intelligence": [
        "machine learning", "deep learning", "nlp", "computer vision",
        "tensorflow", "pytorch", "scikit-learn", "neural network",
        "reinforcement learning", "generative ai",
    ],

    # ── Python ───────────────────────────────────────────────────
    "python": [
        "flask", "django", "fastapi", "pandas", "numpy",
        "scikit-learn", "tensorflow", "pytorch", "scripting",
        "automation", "rest api", "sqlalchemy", "celery",
        "pytest", "jupyter",
    ],

    # ── Java ─────────────────────────────────────────────────────
    "java": [
        "spring", "spring boot", "hibernate", "maven", "gradle",
        "j2ee", "jsp", "servlets", "junit", "microservices",
        "android", "kotlin", "jvm",
    ],

    # ── JavaScript ───────────────────────────────────────────────
    "javascript": [
        "react", "angular", "vue", "node.js", "nodejs",
        "express", "next.js", "typescript", "jquery",
        "webpack", "babel", "npm", "rest api",
    ],

    # ── Software Development / Engineering ───────────────────────
    "software development": [
        "coding", "programming", "debugging", "software testing",
        "version control", "git", "agile", "scrum",
        "object oriented", "oop", "design patterns",
        "rest api", "microservices", "documentation",
    ],

    "software testing": [
        "unit testing", "integration testing", "test automation",
        "selenium", "junit", "pytest", "qa", "quality assurance",
        "test cases", "debugging", "manual testing",
    ],

    "coding": [
        "programming", "software development", "debugging",
        "algorithms", "data structures", "object oriented",
    ],

    "debugging": [
        "software testing", "troubleshooting", "qa",
        "test cases", "unit testing",
    ],

    # ── C++ ──────────────────────────────────────────────────────
    "c++": [
        "object oriented", "oop", "data structures", "algorithms",
        "memory management", "stl", "c", "embedded systems",
        "game development", "system programming",
    ],

    "c#": [
        ".net", "asp.net", "unity", "xamarin", "visual studio",
        "object oriented", "wpf", "mvc",
    ],

    # ── Web Development ──────────────────────────────────────────
    "full stack": [
        "html", "css", "javascript", "react", "node.js",
        "sql", "rest api", "git", "frontend", "backend",
    ],

    "html": ["css", "javascript", "bootstrap", "responsive design", "frontend"],
    "css":  ["html", "sass", "bootstrap", "responsive design", "frontend"],

    "frontend": [
        "html", "css", "javascript", "react", "angular", "vue",
        "bootstrap", "responsive design", "ui", "ux",
    ],

    "backend": [
        "python", "java", "node.js", "sql", "rest api",
        "databases", "server", "api development", "microservices",
    ],

    # ── Cloud / DevOps ───────────────────────────────────────────
    "devops": [
        "docker", "kubernetes", "aws", "azure", "gcp",
        "ci/cd", "jenkins", "terraform", "ansible", "linux",
        "git", "monitoring", "infrastructure as code",
    ],

    "aws": [
        "ec2", "s3", "lambda", "cloudwatch", "rds", "iam",
        "cloud", "serverless", "devops",
    ],

    "cloud": [
        "aws", "azure", "gcp", "serverless", "microservices",
        "docker", "kubernetes", "devops", "infrastructure",
    ],

    "docker": ["kubernetes", "containerization", "devops", "microservices"],
    "kubernetes": ["docker", "containerization", "devops", "orchestration"],

    # ── Database ─────────────────────────────────────────────────
    "sql": [
        "postgresql", "mysql", "sqlite", "database",
        "queries", "stored procedures", "database design",
    ],

    "database": [
        "sql", "postgresql", "mysql", "mongodb", "redis",
        "nosql", "database design", "normalization",
    ],

    # ── Big Data ─────────────────────────────────────────────────
    "big data": [
        "hadoop", "spark", "hive", "kafka", "mapreduce",
        "hdfs", "yarn", "data pipeline", "etl",
    ],

    "hadoop": ["hive", "mapreduce", "hdfs", "yarn", "big data", "spark"],
    "spark":  ["hadoop", "pyspark", "big data", "kafka", "etl"],

    # ── Security ─────────────────────────────────────────────────
    "cybersecurity": [
        "network security", "ethical hacking", "penetration testing",
        "firewalls", "siem", "vulnerability assessment",
        "intrusion detection", "encryption", "risk assessment",
    ],

    "network security": [
        "firewalls", "vpn", "intrusion detection", "siem",
        "cybersecurity", "encryption", "network monitoring",
    ],

    "ethical hacking": [
        "penetration testing", "vulnerability assessment",
        "nmap", "metasploit", "wireshark", "kali linux",
    ],

    # ── Design ───────────────────────────────────────────────────
    "ui/ux": [
        "figma", "adobe xd", "wireframing", "prototyping",
        "user research", "usability testing", "interaction design",
    ],

    "figma": ["wireframing", "prototyping", "ui design", "mockup", "adobe xd"],

    "graphic design": [
        "photoshop", "illustrator", "indesign", "canva",
        "branding", "typography", "logo design", "adobe",
    ],

    # ── Marketing ────────────────────────────────────────────────
    "digital marketing": [
        "seo", "sem", "google ads", "social media marketing",
        "content marketing", "email marketing", "analytics",
        "facebook ads", "instagram", "crm", "hubspot",
    ],

    "seo": [
        "sem", "keyword research", "google analytics",
        "content marketing", "on-page seo", "backlinks",
    ],

    # ── Project Management ───────────────────────────────────────
    "project management": [
        "agile", "scrum", "kanban", "jira", "confluence",
        "stakeholder management", "planning", "risk management",
        "budget management", "team leadership",
    ],

    "agile": ["scrum", "kanban", "sprint", "jira", "retrospective"],
    "scrum": ["agile", "sprint", "jira", "product backlog", "standup"],

    # ── HR / Finance ─────────────────────────────────────────────
    "human resources": [
        "recruitment", "payroll", "employee relations",
        "onboarding", "hr policies", "performance appraisal",
        "talent management", "hrms",
    ],

    "recruitment": [
        "hiring", "screening", "interviewing", "talent acquisition",
        "sourcing", "linkedin", "job portals",
    ],

    "accounting": [
        "tally", "gst", "taxation", "bookkeeping",
        "financial statements", "audit", "compliance",
    ],

    "financial analysis": [
        "excel", "financial modeling", "budgeting", "forecasting",
        "variance analysis", "ratio analysis", "valuation",
    ],

    # ── Mobile ───────────────────────────────────────────────────
    "mobile development": [
        "ios", "android", "swift", "kotlin", "react native",
        "flutter", "mobile application", "app development",
    ],

    "ios": ["swift", "objective-c", "xcode", "cocoa", "mobile application"],
    "android": ["kotlin", "java", "android studio", "mobile application"],

    # ── Embedded / Hardware ──────────────────────────────────────
    "embedded systems": [
        "c", "c++", "microcontroller", "arduino", "raspberry pi",
        "rtos", "firmware", "iot",
    ],
}


# ================================================================
# REVERSE TAXONOMY — child skill → parent skills
# If resume has "TensorFlow", mark "Machine Learning" as known
# ================================================================

def build_reverse_taxonomy():
    reverse = {}
    for parent, children in SKILL_TAXONOMY.items():
        for child in children:
            if child not in reverse:
                reverse[child] = []
            reverse[child].append(parent)
    return reverse

REVERSE_TAXONOMY = build_reverse_taxonomy()


# ================================================================
# EXPAND SKILLS
# Given a set of skills, return expanded set including implied skills
# ================================================================

def expand_skills(skill_set):
    """
    Expand a set of skills using the taxonomy.

    Example:
        Input:  {"machine learning", "python", "c++"}
        Output: {"machine learning", "python", "c++",
                 "scikit-learn", "tensorflow", "deep learning",
                 "flask", "django", "pandas", ...}
    """
    expanded = set(skill_set)

    for skill in list(skill_set):
        # Forward: parent → children
        if skill in SKILL_TAXONOMY:
            expanded.update(SKILL_TAXONOMY[skill])

        # Reverse: child → parents
        if skill in REVERSE_TAXONOMY:
            expanded.update(REVERSE_TAXONOMY[skill])

    return expanded


# ================================================================
# EXTRACT SKILLS FROM TEXT
# ================================================================

SKILL_LIST = list(SKILL_TAXONOMY.keys()) + [
    child for children in SKILL_TAXONOMY.values() for child in children
]
SKILL_LIST = list(set(SKILL_LIST))
# Sort longest first so multi-word skills match before single words
SKILL_LIST.sort(key=len, reverse=True)


def extract_skills_from_text(text):
    """
    Extract all skills found in text using keyword matching + spaCy NER.
    Returns a set of skill strings.
    """
    text_lower = text.lower()
    found = set()

    # Keyword matching
    for skill in SKILL_LIST:
        if skill in text_lower:
            found.add(skill)

    # spaCy NER for extra tool/product names
    if SPACY_AVAILABLE:
        try:
            doc = nlp(text[:50000])
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT"]:
                    found.add(ent.text.lower().strip())
        except Exception:
            pass

    return found


# ================================================================
# MAIN MATCHING FUNCTION
# ================================================================

def calculate_match(resume_text, job_description):
    """
    Semantic hybrid matching:

    Step 1 — Extract skills from resume
    Step 2 — Expand resume skills using taxonomy
             (Machine Learning → implies TensorFlow, Scikit-learn, etc.)
    Step 3 — Compare expanded resume skills against job requirements
    Step 4 — Skill score (70%) + TF-IDF cosine score (30%)
    Step 5 — Generate recommendation
    """

    if not resume_text or not job_description:
        return 0.0, "", "", "Unable to analyze."

    # ── Step 1: Extract skills directly from resume ──────────────
    raw_resume_skills = extract_skills_from_text(resume_text)

    # ── Step 2: Expand using taxonomy ───────────────────────────
    # e.g. "machine learning" in resume → also marks tensorflow, sklearn, etc.
    expanded_resume_skills = expand_skills(raw_resume_skills)

    resume_lower = resume_text.lower()

    # ── Step 3: Match job skills against expanded resume skills ──
    job_skills = [s.strip().lower() for s in job_description.split(",") if s.strip()]

    if not job_skills:
        return 0.0, "", "", "No skills listed in job description."

    matched_skills = []
    missing_skills = []

    for skill in job_skills:
        skill_lower = skill.lower()

        # Check 1: direct text match in resume
        direct_match = skill_lower in resume_lower

        # Check 2: in expanded skills (taxonomy inference)
        taxonomy_match = skill_lower in expanded_resume_skills

        # Check 3: any word of the skill appears in resume (partial)
        words = skill_lower.split()
        partial_match = any(w in resume_lower for w in words if len(w) > 3)

        if direct_match or taxonomy_match or partial_match:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    # ── Step 4: Calculate Scores ─────────────────────────────────
    skill_score = (len(matched_skills) / len(job_skills)) * 100

    # TF-IDF cosine similarity
    tfidf_score = 0.0
    try:
        vectorizer = TfidfVectorizer(min_df=1)
        matrix = vectorizer.fit_transform([resume_lower, job_description.lower()])
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])
        tfidf_score = float(similarity[0][0]) * 100
    except Exception:
        pass

    # Weighted final score
    # Weighted final score
    if len(missing_skills) == 0:
        # All skills matched — score is 100% regardless of TF-IDF
        final_score = 100.0
    elif len(job_skills) < 4:
        final_score = skill_score
    else:
        final_score = round((skill_score * 0.7) + (tfidf_score * 0.3), 2)

    final_score = round(min(final_score, 100.0), 2)

    # ── Step 5: Recommendation ───────────────────────────────────
    matched_count = len(matched_skills)
    total_count   = len(job_skills)

    if final_score >= 75:
        recommendation = (
            f"Strong Fit ✅ — Excellent match! {matched_count}/{total_count} required skills found. "
            f"Highly recommended for this role."
        )
    elif final_score >= 50:
        recommendation = (
            f"Moderate Fit 🟡 — Good potential. {matched_count}/{total_count} skills matched. "
            f"Some skill gaps exist but candidate shows strong promise."
        )
    elif final_score >= 25:
        recommendation = (
            f"Weak Fit ❌ — Limited match. Only {matched_count}/{total_count} skills found. "
            f"Candidate should build missing skills before applying."
        )
    else:
        recommendation = (
            f"Poor Fit ❌ — Very limited skill overlap. {matched_count}/{total_count} skills found. "
            f"Resume does not align with this job's requirements."
        )

    return (
        final_score,
        ", ".join(matched_skills),
        ", ".join(missing_skills),
        recommendation
    )