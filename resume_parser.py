"""
resume_parser.py
================
Handles:
1. PDF extraction using pdfminer.six (more accurate than PyPDF2)
2. DOCX extraction using python-docx
3. Text cleaning using NLTK (stopwords, lemmatization)
"""

import re
import io
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (only first time)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)


# ===============================
# PDF EXTRACTION (pdfminer.six)
# ===============================

def read_pdf(file):
    """
    Extract text from PDF using pdfminer.six.
    More accurate than PyPDF2 for complex layouts, columns, tables.
    """
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract

        if hasattr(file, 'read'):
            file_bytes = file.read()
        else:
            file_bytes = file

        text = pdfminer_extract(io.BytesIO(file_bytes))
        return text if text and text.strip() else ""

    except ImportError:
        # Fallback to PyPDF2 if pdfminer not installed
        print("[resume_parser] pdfminer.six not found, falling back to PyPDF2")
        return _read_pdf_pypdf2(file)
    except Exception as e:
        print(f"[resume_parser] PDF error: {e}")
        return ""


def _read_pdf_pypdf2(file):
    """Fallback PDF reader using PyPDF2"""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        print(f"[resume_parser] PyPDF2 error: {e}")
        return ""


# ===============================
# DOCX EXTRACTION
# ===============================

def read_docx(file):
    """Extract text from DOCX (Word) file using python-docx."""
    try:
        import docx
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[resume_parser] DOCX error: {e}")
        return ""


# ===============================
# TEXT CLEANING
# ===============================

def clean_text(text):
    """
    Clean and preprocess resume text for ML/NLP.

    Steps:
    1. Lowercase
    2. Remove URLs, emails, phone numbers
    3. Remove special characters
    4. Remove stopwords
    5. Lemmatize (running→run, developer→develop)
    """
    if not text or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)

    # Remove emails
    text = re.sub(r'\S+@\S+', ' ', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', ' ', text)

    # Remove special characters — keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update({
        'resume', 'curriculum', 'vitae', 'cv', 'name',
        'address', 'phone', 'email', 'objective', 'summary',
        'profile', 'references', 'available', 'request'
    })

    words = [w for w in text.split() if w not in stop_words and len(w) > 2]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]

    return ' '.join(words)


# ===============================
# QUICK TEST
# ===============================

if __name__ == "__main__":
    sample = "Hi! I'm a Python Developer. Email: dev@gmail.com | Phone: 9876543210. Skills: Python, Flask, REST API, PostgreSQL, Docker."
    print("Original:", sample)
    print("Cleaned: ", clean_text(sample))