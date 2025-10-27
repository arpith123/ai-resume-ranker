# utils.py - Shared utility functions for resume analysis

from pdfminer.high_level import extract_text
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy once
nlp = spacy.load("en_core_web_sm")

def parse_pdf_file(file_path):
    """Extract text from PDF file path."""
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
    """Enhanced keyword extraction from job description."""
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
    """Extract individual skills from text."""
    skills = skills_text.lower().replace('\n', ',').replace('•', ',').replace('|', ',').replace('—', ',').split(',')
    return set([s.strip() for s in skills if s.strip() and len(s.strip()) > 2])

def get_match_label(score):
    """Return match strength label."""
    if score >= 0.4:
        return "Strong Match"
    elif score >= 0.3:
        return "Good Match"
    elif score >= 0.2:
        return "Fair Match"
    else:
        return "Weak Match"

def get_score_color(score):
    """Return emoji indicator based on score."""
    if score >= 0.4:
        return "🟢"
    elif score >= 0.3:
        return "🟡"
    else:
        return "🟠"

def analyze_resume(resume_text, jd_text, jd_keywords):
    """Complete analysis of a single resume against JD."""
    sections = extract_sections(resume_text)
    resume_for_scoring = " ".join([sections.get(s, '') for s in ["Skills", "Profile", "Experience"]])
    similarity_score = compute_similarity(resume_for_scoring, jd_text)
    resume_skills = extract_skills_set(sections.get('Skills', ''))
    matched_keywords = resume_skills & jd_keywords
    
    return {
        'similarity_score': similarity_score,
        'resume_skills_count': len(resume_skills),
        'matched_keywords_count': len(matched_keywords),
        'matched_keywords': sorted(list(matched_keywords)),
        'sections': sections,
        'match_label': get_match_label(similarity_score),
        'score_indicator': get_score_color(similarity_score),
        'full_text': resume_text  # Add full text for LLM
    }
