import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import PyPDF2
from io import BytesIO

# --- Page configuration ---
st.set_page_config(page_title="AI Resume Screener V3 (Gemini)", page_icon="🚀", layout="wide")

# --- Custom CSS for UI ---
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
    [data-testid="stSidebar"] { background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(10px); }
    .glass-card { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
    .stButton > button { background: linear-gradient(135deg, #3b82f6, #8b5cf6); color: white; border: none; font-weight: 600; border-radius: 8px; transition: all 0.3s; }
    .skill-tag { padding: 0.3rem 0.8rem; border-radius: 6px; display: inline-block; margin: 0.2rem; font-size: 0.85rem; font-weight: 500; }
    .matched { background: rgba(34, 197, 94, 0.2); border: 1px solid rgba(34, 197, 94, 0.4); color: #86efac; }
    .missing { background: rgba(239, 68, 68, 0.2); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; }
    .feedback-box { background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; padding: 1rem; }
    h1, h2, h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# --- API Configuration ---
API_BASE = 'http://localhost:8000'

# --- Functions ---
def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def analyze_resumes(job_title, groq_key, gemini_key, jd, skills, files, top_n):
    """Sends the request to the FastAPI backend."""
    try:
        form_data = {
            'job_title': job_title,
            'groq_api_key': groq_key,
            'gemini_api_key': gemini_key,
            'job_description': jd,
            'required_skills': skills,
            'top_n': top_n
        }
        file_data = [('files', (f.name, f.getvalue(), f.type)) for f in files]
        res = requests.post(f"{API_BASE}/rank-resumes", data=form_data, files=file_data, timeout=300)

        if res.status_code == 200:
            return res.json()
        else:
            st.error(f"API Error: {res.status_code} - {res.text}")
            return None
            
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return None

def to_csv(results):
    """Converts the results dictionary to a CSV file for download."""
    data = [{
        'Rank': c['rank'], 'Filename': c['filename'],
        'Semantic Score': f"{c['semantic_score']:.2%}",
        'Skill Matches': f"{c['skill_match_count']}/{c['total_skills_required']}",
        'Matched Skills': ', '.join(c['matched_skills']),
        'Missing Skills': ', '.join(c['missing_skills']),
        'AI Verdict': c['ai_verdict'], 'AI Feedback': c['ai_feedback']
    } for c in results['top_candidates']]
    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("<h2>AI Resume Screener</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 1. Job Title")
    job_title = st.text_input("Job Title", "Senior Data Scientist", label_visibility="collapsed")
    
    st.markdown("### 2. Groq API Key (for Feedback)")
    # Restored dropdown for Groq API Key
    groq_api_keys = [
        "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", # Placeholder
        "your-actual-key-1",
        "your-actual-key-2"
    ]
    groq_key = st.selectbox("Groq API Key", options=groq_api_keys, label_visibility="collapsed")
    
    st.markdown("### 3. Gemini API Key (for Embeddings)")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
    
    st.markdown("### 4. Job Description")
    # Restored tabs for Job Description input
    jd_tab1, jd_tab2 = st.tabs(["Enter Manually", "Upload PDF"])
    
    with jd_tab1:
        jd_text = st.text_area("Job Description Text", "Seeking a Senior Data Scientist. Skills: Python, SQL, ML, TensorFlow, AWS.", height=150, label_visibility="collapsed")
    with jd_tab2:
        jd_file = st.file_uploader("Upload JD PDF", type=['pdf'], label_visibility="collapsed")

    st.markdown("### 5. Required Skills")
    skills = st.text_input("Required Skills (comma-separated)", "Python, SQL, Machine Learning, TensorFlow, AWS")

    st.markdown("### 6. Upload Resumes")
    files = st.file_uploader("Upload Resumes", accept_multiple_files=True, type=['pdf', 'docx', 'txt'])
    
    st.markdown("### 7. Options")
    top_n = st.number_input("Number of candidates to show", min_value=1, max_value=50, value=10)

    st.markdown("---")
    analyze_btn = st.button("🚀 Start Analysis", use_container_width=True)

# --- Main Content Area ---
st.markdown("# 📊 Analysis Results")

if 'results' not in st.session_state:
    st.session_state.results = None

if analyze_btn:
    # Determine the job description source
    final_jd = ""
    if jd_file:
        final_jd = extract_text_from_pdf(jd_file.getvalue())
    else:
        final_jd = jd_text

    if all([job_title, gemini_key, final_jd, skills, files]):
        with st.spinner("Analyzing resumes... This may take a moment."):
            results = analyze_resumes(job_title, groq_key, gemini_key, final_jd, skills, files, top_n)
            st.session_state.results = results
    else:
        st.error("Please fill all fields (including a JD) and upload at least one resume.")

if st.session_state.results:
    res = st.session_state.results
    st.success(f"Analysis complete! Processed {res['total_resumes_processed']} resumes in {res['processing_time']:.1f}s.")
    
    st.download_button("📥 Export Results", to_csv(res), f"analysis_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    
    for c in res['top_candidates']:
        st.markdown(f"""
        <div class="glass-card">
            <h3><span style='color: #94a3b8;'>#{c['rank']}</span> {c['filename']}</h3>
            <div class="feedback-box">
                <p><b>AI Verdict:</b> {c['ai_verdict']}</p>
                <p><b>AI Feedback:</b> {c['ai_feedback']}</p>
            </div>
            <br>
            <p><b>Key Evidence:</b> <i>{c['key_matching_chunk']}</i></p>
            <p><b>Matched Skills ({c['skill_match_count']}):</b> {''.join(f"<span class='skill-tag matched'>{s}</span>" for s in c['matched_skills'])}</p>
            <p><b>Missing Skills ({len(c['missing_skills'])}):</b> {''.join(f"<span class='skill-tag missing'>{s}</span>" for s in c['missing_skills'])}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("Fill in the details on the left, upload resumes, and click 'Start Analysis' to begin.")

