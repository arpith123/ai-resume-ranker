# 🎯 AI-Powered Resume Ranker

**AI-Powered Resume Ranker** is a modern, production-ready tool for ranking multiple resumes against any job description using NLP, machine learning, and LLM-based Q&A—complete with a beautiful Streamlit web UI and FastAPI backend.

---

## 🚀 Features

- **Batch Resume Ranking:** Upload multiple PDF resumes, analyze and rank them against a job description.
- **Skill & Section Extraction:** Extracts skills, experience, and profile data from resumes using NLP.
- **Automated JD Parsing:** Identifies key requirements from any pasted or uploaded job description.
- **Scoring & Similarity:** Calculates fit based on advanced text similarity and direct skill matching.
- **Interactive Web UI:** Visual, sortable results with CSV export and detailed candidate breakdown.
- **LLM Q&A (RAG):** Ask natural language questions about any candidate's resume (ChatGPT-like answers).
- **REST API:** FastAPI backend allows programmatic and integration use.
- **Modern Design:** Supports dark mode and expandable sections for clarity.

---

## 🛠️ Requirements

- Python 3.9+
- Required packages (install using pip):
  - streamlit
  - fastapi
  - uvicorn
  - pdfminer.six
  - python-docx
  - scikit-learn
  - spacy
  - pandas
  - huggingface-hub
  - openai (optional, if using OpenAI for LLM)

- Download spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

**Tip:** Install everything at once:
```bash
pip install -r requirements.txt
```

---

## 💡 Usage (For Recruiters, HR, Students)

### 1. **Streamlit Web App (UI)**

```bash
streamlit run app_improved.py
```

- Open `http://localhost:8501` in your browser.
- **Step 1:** Enter or upload a job description in the sidebar.
- **Step 2:** Upload one or more resumes (PDF).
- **Step 3:** Click "Analyze Resumes".
- View ranked results, skill match details, and expand each candidate for more insights.
- Use the "🤖 Enable AI Q&A" checkbox in each candidate's detailed section to ask LLM-powered questions about that resume.
- Download ranking or detailed report as CSV for further use.

---

### 2. **API Access (Batch/Integration Use)**

Start FastAPI server:
```bash
python api.py
```

- Open `http://localhost:8000/docs` for interactive Swagger UI.
- Use `/rank` to POST a job description and one or more PDF resumes. Returns JSON ranking, scores, and matches.
- Example with curl:
```bash
curl -X POST "http://localhost:8000/rank" \
  -F "jd=Python, SQL, Azure required" \
  -F "resumes=@sample_resume1.pdf"
```

---

### 3. **LLM "Ask the Resume" Feature**

Within the Streamlit UI, in each detailed candidate section:
- Tick `🤖 Enable AI Q&A`
- Type a question (e.g., "List ML projects from this resume.")
- Choose provider (HuggingFace Free, or OpenAI with your API key)
- Get AI-powered answer instantly!

---

## 📂 Project Structure

```
resume_ranker/
├─ app_improved.py          # Streamlit UI app
├─ api.py                   # FastAPI REST API backend
├─ llm_integration.py       # LLM Q&A functions
├─ utils.py                 # Shared utilities
├─ requirements.txt         # Python dependencies
├─ sample_resume1.pdf       # Example resume(s)
├─ jd.txt                   # Example job description
└─ README.md                # This file
```

---

## ⚠️ Limitations

- PDF resumes only (for now). DOCX parsing can be added.
- LLM Q&A (HuggingFace) is rate-limited for free users; OpenAI API requires your own key.
- Designed/tested for resumes in English.

---

## 🙏 Credits

- spaCy NLP, scikit-learn
- PDFMiner, HuggingFace Hub, OpenAI GPT
- Streamlit, FastAPI, Pandas

---

## 📄 License

MIT License (see LICENSE file)

---

## ✨ Author

- [Arpith](https://github.com/arpith123)
- `arpith.pr@gmail.com`

---

**Happy Ranking! Built with ❤️ for the Future of Work.**
