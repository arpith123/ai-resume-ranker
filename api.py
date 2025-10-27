# api.py - FastAPI Backend for Resume Ranker

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import uvicorn

# Import our existing functions
from pdfminer.high_level import extract_text
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import os

# Load spaCy
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("spaCy model loaded successfully!")

# ============ UTILITY FUNCTIONS (from previous code) ============

def parse_pdf_file(file_path):
    """Extract text from PDF file."""
    try:
        return extract_text(file_path)
    except Exception as e:
        return ""

def extract_sections(text):
    """Extract sections from resume text."""
    headings = ['PROFILE', 'SKILLS', 'EXPERIENCE', 'PROJECT', 'EDUCATION', 'CERTIFICATIONS']
    pattern = r'^(%s)[\s:]*$' % '|'.join(headings)
    matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
    sections = {}
    for i, match in enumerate(matches):
        section_name = match.group(1).capitalize()
        start = match.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        section_content = text[start:end].strip()
        if section_content:
            sections[section_name] = section_content
    return sections

def extract_keywords_from_jd(jd_text):
    """Extract keywords from job description."""
    doc = nlp(jd_text)
    keywords = set([chunk.text.lower() for chunk in doc.noun_chunks])
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
            keywords.add(token.text.lower())
    return keywords

def compute_similarity(text1, text2):
    """Compute TF-IDF cosine similarity."""
    if not text1.strip() or not text2.strip():
        return 0.0
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score, 3)

def extract_skills_set(skills_text):
    """Extract skills from text."""
    skills = skills_text.lower().replace('\n', ',').replace('•', ',').replace('|', ',').replace('—', ',').split(',')
    return set([s.strip() for s in skills if s.strip() and len(s.strip()) > 2])

def analyze_resume(resume_text, jd_text, jd_keywords):
    """Analyze resume against job description."""
    sections = extract_sections(resume_text)
    resume_for_scoring = " ".join([sections.get(s, '') for s in ["Skills", "Profile", "Experience"]])
    similarity_score = compute_similarity(resume_for_scoring, jd_text)
    resume_skills = extract_skills_set(sections.get('Skills', ''))
    matched_keywords = resume_skills & jd_keywords
    
    return {
        'similarity_score': similarity_score,
        'matched_keywords_count': len(matched_keywords),
        'matched_keywords': sorted(list(matched_keywords)),
        'resume_skills_count': len(resume_skills),
        'sections': sections
    }

# ============ PYDANTIC MODELS ============

class RankingResult(BaseModel):
    rank: int
    filename: str
    similarity_score: float
    matched_keywords_count: int
    matched_keywords: List[str]
    resume_skills_count: int

class RankingResponse(BaseModel):
    total_resumes: int
    rankings: List[RankingResult]
    jd_keywords_count: int

# ============ FASTAPI APP ============

print("Creating FastAPI app...")
app = FastAPI(
    title="AI-Powered Resume Ranker API",
    description="API for ranking resumes against job descriptions using NLP and ML",
    version="1.0.0"
)
print("FastAPI app created successfully!")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "AI-Powered Resume Ranker API",
        "version": "1.0.0",
        "endpoints": {
            "/rank": "POST - Rank multiple resumes against a job description",
            "/health": "GET - Health check endpoint",
            "/docs": "GET - Interactive API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "resume-ranker-api"}

@app.post("/rank", response_model=RankingResponse)
async def rank_resumes(
    jd: str = Form(..., description="Job description text"),
    resumes: List[UploadFile] = File(..., description="List of resume PDF files")
):
    """
    Rank multiple resumes against a job description.
    
    - **jd**: Job description as text
    - **resumes**: List of PDF resume files to rank
    
    Returns ranked list of candidates with similarity scores and matched keywords.
    """
    
    if not jd.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    
    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume file is required")
    
    # Extract JD keywords
    jd_keywords = extract_keywords_from_jd(jd)
    
    results = []
    
    # Process each resume
    for resume_file in resumes:
        if not resume_file.filename.endswith('.pdf'):
            continue
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await resume_file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Parse and analyze
        resume_text = parse_pdf_file(tmp_path)
        os.unlink(tmp_path)
        
        if resume_text:
            analysis = analyze_resume(resume_text, jd, jd_keywords)
            analysis['filename'] = resume_file.filename
            results.append(analysis)
    
    # Sort by similarity score
    results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
    
    # Format response
    rankings = [
        RankingResult(
            rank=i+1,
            filename=r['filename'],
            similarity_score=r['similarity_score'],
            matched_keywords_count=r['matched_keywords_count'],
            matched_keywords=r['matched_keywords'][:20],  # Limit to top 20
            resume_skills_count=r['resume_skills_count']
        )
        for i, r in enumerate(results)
    ]
    
    return RankingResponse(
        total_resumes=len(results),
        rankings=rankings,
        jd_keywords_count=len(jd_keywords)
    )

# ============ RUN SERVER ============

print("Script loaded successfully!")
if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
