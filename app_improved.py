# app_improved.py - Enhanced Streamlit UI with LLM Integration

import streamlit as st
from pdfminer.high_level import extract_text
import pandas as pd
import tempfile
import os

# Import shared utilities
from utils import (
    extract_sections, extract_keywords_from_jd, 
    analyze_resume, get_score_color, get_match_label
)

# Import LLM integration
from llm_integration import add_llm_qa_to_streamlit

# ============ HELPER FUNCTIONS ============

def parse_pdf_bytes(file_bytes):
    """Extract text from uploaded PDF bytes."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name
        text = extract_text(tmp_path)
        os.unlink(tmp_path)
        return text
    except Exception as e:
        st.error(f"PDF parsing error: {e}")
        return ""

# ============ STREAMLIT UI ============

st.set_page_config(page_title="AI Resume Ranker", page_icon="🎯", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .matched-keyword {
        display: inline-block;
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🎯 AI-Powered Resume Ranker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload resumes and job description to rank candidates with AI-powered analysis</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📋 Job Description")
    jd_input_method = st.radio("Input Method:", ["Paste Text", "Upload File"])
    
    if jd_input_method == "Paste Text":
        jd_text = st.text_area("Paste Job Description:", height=300)
    else:
        jd_file = st.file_uploader("Upload JD (.txt)", type=['txt'])
        jd_text = jd_file.read().decode('utf-8') if jd_file else ""
    
    st.markdown("---")
    st.markdown("**📖 Instructions:**")
    st.markdown("1. Enter/upload job description")
    st.markdown("2. Upload resume(s) in PDF")
    st.markdown("3. Analyze and view rankings")
    st.markdown("4. Use AI Q&A for insights")
    
    st.markdown("---")
    st.markdown("**ℹ️ Scoring:**")
    st.markdown("🟢 Strong (40%+)")
    st.markdown("🟡 Good (30-40%)")
    st.markdown("🟠 Fair (<30%)")

# Main area
st.header("📤 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload one or more resumes (PDF)",
    type=['pdf'],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} resume(s) uploaded")

# Initialize session state for results
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Analyze button
if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
    if not jd_text:
        st.error("⚠️ Please provide a job description!")
    elif not uploaded_files:
        st.error("⚠️ Please upload at least one resume!")
    else:
        jd_keywords = extract_keywords_from_jd(jd_text)
        
        with st.spinner("🔍 Analyzing..."):
            results = []
            progress = st.progress(0)
            
            for idx, file in enumerate(uploaded_files):
                resume_bytes = file.read()
                resume_text = parse_pdf_bytes(resume_bytes)
                
                if resume_text:
                    analysis = analyze_resume(resume_text, jd_text, jd_keywords)
                    analysis['filename'] = file.name
                    analysis['full_text'] = resume_text  # Store for LLM
                    results.append(analysis)
                
                progress.progress((idx + 1) / len(uploaded_files))
            
            progress.empty()
            results = sorted(results, key=lambda x: x['similarity_score'], reverse=True)
            
            # Store results in session state
            st.session_state.analysis_results = results
            
            st.success(f"✅ Analyzed {len(results)} resume(s)")

# Display results if available in session state
if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Summary metrics
    st.header("📊 Ranking Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Candidates", len(results))
    with col2:
        avg = sum([r['similarity_score'] for r in results]) / len(results)
        st.metric("Average Score", f"{avg:.1%}")
    with col3:
        strong = sum([1 for r in results if r['similarity_score'] >= 0.4])
        st.metric("Strong Matches", strong)
    
    # Rankings table
    df = pd.DataFrame([{
        'Rank': i+1,
        'Candidate': r['filename'],
        'Match': f"{r['score_indicator']} {r['match_label']}",
        'Score': f"{r['similarity_score']:.1%}",
        'Skills Matched': r['matched_keywords_count'],
        'Total Skills': r['resume_skills_count']
    } for i, r in enumerate(results)])
    
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Detailed analysis
    st.header("🔍 Detailed Analysis")
    
    for i, result in enumerate(results):
        with st.expander(
            f"{result['score_indicator']} #{i+1} - {result['filename']} ({result['match_label']})",
            expanded=(i==0)
        ):
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Score", f"{result['similarity_score']:.1%}")
            with col2:
                st.metric("Quality", result['match_label'])
            with col3:
                st.metric("Matched", result['matched_keywords_count'])
            with col4:
                st.metric("Total Skills", result['resume_skills_count'])
            
            # Explanation
            st.markdown("---")
            st.markdown("**💡 Analysis:**")
            exp = f"**{result['matched_keywords_count']} skills matched**, "
            if result['matched_keywords']:
                exp += f"including: **{', '.join(result['matched_keywords'][:5])}**. "
            exp += f"Overall **{result['similarity_score']:.1%}** similarity = **{result['match_label'].lower()}**."
            
            if result['similarity_score'] >= 0.3:
                st.success(exp)
            else:
                st.info(exp)
            
            # Keywords
            if result['matched_keywords']:
                st.markdown("**🎯 Matched Keywords:**")
                kw_html = "".join([f'<span class="matched-keyword">{k}</span>' 
                                  for k in result['matched_keywords'][:15]])
                st.markdown(kw_html, unsafe_allow_html=True)
            
            # Sections
            st.markdown("---")
            st.markdown("**📄 Resume Sections:**")
            for sec, content in result['sections'].items():
                with st.expander(f"📌 {sec}"):
                    st.text(content)
            
            # LLM Q&A Integration
            st.markdown("---")
            st.markdown("### 🤖 AI Q&A Assistant")
            st.markdown("*Ask any question about this candidate's resume and get AI-powered insights*")
            
            if st.checkbox("Enable AI Q&A Assistant", key=f"llm_checkbox_{i}"):
                full_text = "\n\n".join([f"{k}:\n{v}" 
                                        for k, v in result['sections'].items()])
                add_llm_qa_to_streamlit(full_text, result['filename'])
    
    # Export
    st.header("💾 Export Results")
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Rankings (CSV)",
        csv,
        "rankings.csv",
        "text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ | AI-Powered Resume Ranker © 2025</p>
    <p style='font-size: 0.8rem;'>Powered by spaCy, scikit-learn, HuggingFace</p>
</div>
""", unsafe_allow_html=True)
