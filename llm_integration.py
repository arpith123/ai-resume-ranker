# llm_integration.py - RAG/LLM Integration for Resume Q&A

import os
from typing import Optional
import streamlit as st

# Export the main function for Streamlit integration
__all__ = ['add_llm_qa_to_streamlit']

# Free LLM options (no API key needed for HuggingFace Inference API - limited free tier)
from huggingface_hub import InferenceClient

# ============ LLM SETUP ============

class ResumeLLM:
    """Simple RAG implementation using HuggingFace free inference."""
    
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize LLM client.
        Using Mistral-7B-Instruct (free via HuggingFace Inference API)
        No API key needed for basic usage (rate-limited)
        """
        self.client = InferenceClient(model=model_name)
        
    def ask_resume(self, resume_text: str, question: str, max_tokens: int = 300) -> str:
        """
        Ask a question about a resume using LLM with RAG approach.
        
        Args:
            resume_text: Full resume content
            question: User's question
            max_tokens: Maximum response length
        
        Returns:
            LLM-generated answer
        """
        
        # Simple pattern-based answers (fallback when LLM is not available)
        resume_lower = resume_text.lower()
        question_lower = question.lower()
        
        # Try to find simple answers from the resume text
        if "skill" in question_lower or "expertise" in question_lower:
            # Extract skills section if available
            if "skills" in resume_lower:
                skills_start = max(0, resume_lower.find("skills"))
                skills_end = skills_start + 500
                return "Based on the resume, the candidate's skills include: " + \
                       resume_text[skills_start:skills_end]
        
        # Create RAG-style prompt
        prompt = f"""You are an AI assistant analyzing a candidate's resume. Answer the question based ONLY on the information provided in the resume.

Resume:
{resume_text[:3000]}  

Question: {question}

Answer (be concise and specific, cite resume details):"""
        
        try:
            # Call HuggingFace Inference API (free tier)
            response = self.client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.3,
                do_sample=True
            )
            return response.strip()
        
        except Exception as e:
            # Return a helpful message instead of error
            return f"I apologize, but the AI-powered Q&A feature is currently experiencing connectivity issues with the HuggingFace API.\n\n" \
                   f"**Your Question:** {question}\n\n" \
                   f"Please try:\n" \
                   f"1. Use the OpenAI provider option (requires API key)\n" \
                   f"2. Check the 'Resume Sections' above for detailed information\n" \
                   f"3. Review the 'Matched Keywords' to see relevant skills"

# ============ OPENAI ALTERNATIVE (if user has API key) ============

def ask_resume_openai(resume_text: str, question: str, api_key: str) -> str:
    """
    Alternative using OpenAI API (requires API key with credits).
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert HR assistant analyzing resumes. Answer questions based only on the resume content provided."
                },
                {
                    "role": "user",
                    "content": f"Resume:\n{resume_text[:4000]}\n\nQuestion: {question}"
                }
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        return f"OpenAI API Error: {str(e)}"

# ============ STREAMLIT UI COMPONENT FOR Q&A ============

def add_llm_qa_to_streamlit(resume_text: str, candidate_name: str):
    """
    Add LLM Q&A interface to Streamlit app.
    Call this in your main app.py after displaying candidate details.
    """
    
    st.markdown("---")
    st.markdown(f"### 🤖 Ask Questions About {candidate_name}")
    st.markdown("*Powered by AI - Ask anything about this candidate's resume*")
    
    # Question input
    question = st.text_input(
        "Ask a question about this resume:",
        placeholder="e.g., Does this candidate have experience with Azure?",
        key=f"qa_{candidate_name}"
    )
    
    # LLM provider selection
    llm_provider = st.radio(
        "Choose LLM Provider:",
        ["🆓 HuggingFace (Free)", "🔑 OpenAI (Requires API Key)"],
        horizontal=True
    )
    
    # OpenAI API key input (only show if OpenAI selected)
    if llm_provider.startswith("🔑"):
        api_key = st.text_input(
            "Enter OpenAI API Key:",
            type="password",
            help="Get your API key from platform.openai.com",
            key=f"openai_key_{candidate_name}"
        )
        can_answer = bool(api_key and api_key.strip())
    else:
        api_key = None
        can_answer = True
    
    if st.button("🔍 Get Answer", key=f"btn_{candidate_name}"):
        if not question.strip():
            st.warning("Please enter a question first!")
        elif not can_answer:
            st.warning("Please provide an OpenAI API key to use this provider.")
        else:
            with st.spinner("🤖 AI is thinking..."):
                try:
                    if llm_provider.startswith("🆓"):
                        # Use free HuggingFace
                        llm = ResumeLLM()
                        answer = llm.ask_resume(resume_text, question)
                    else:
                        # Use OpenAI (requires API key)
                        answer = ask_resume_openai(resume_text, question, api_key)
                    
                    st.markdown("**Answer:**")
                    st.success(answer)
                    
                except Exception as e:
                    st.markdown("**Answer:**")
                    st.error(f"Sorry, the AI is having trouble right now. Error: {str(e)}")
                    st.info("💡 You can try selecting the OpenAI provider or check the resume sections for detailed information.")
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        st.markdown("""
        - What is this candidate's main expertise?
        - Does this candidate have cloud platform experience?
        - What machine learning projects has this candidate worked on?
        - What certifications does this candidate have?
        - Summarize this candidate's work experience
        - What programming languages does this candidate know?
        """)
