import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
import json
import base64
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="AI Resume Screener V3",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main theme styling */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(20px);
    }
    
    /* Card styling */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 0 8px 32px rgba(15, 23, 42, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(59, 130, 246, 0.3);
    }
    
    /* Success/Warning/Error badges */
    .skill-matched {
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #86efac;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .skill-missing {
        background: rgba(239, 68, 68, 0.2);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fca5a5;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Progress circle */
    .progress-circle {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        background: conic-gradient(#22c55e var(--progress), #1e293b 0deg);
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
    }
    
    .progress-circle::before {
        content: "";
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: #0f172a;
        position: absolute;
    }
    
    .progress-text {
        position: relative;
        font-size: 1.5rem;
        font-weight: bold;
        color: #e2e8f0;
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(30, 41, 59, 0.3);
        border: 2px dashed #475569;
        border-radius: 8px;
        padding: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e2e8f0 !important;
    }
    
    /* Text */
    .stMarkdown {
        color: #cbd5e1;
    }
    
    /* Evidence text box */
    .evidence-box {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 1rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* AI feedback box */
    .feedback-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    
    .verdict-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .verdict-excellent {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .verdict-good {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    
    .verdict-poor {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

# API Configuration
API_BASE = 'http://localhost:8000'

def create_circular_progress(score):
    """Create a circular progress indicator using HTML/CSS"""
    color = "#22c55e" if score >= 80 else "#f59e0b" if score >= 60 else "#ef4444"
    return f"""
        <div style="display: flex; flex-direction: column; align-items: center;">
            <div class="progress-circle" style="--progress: {score * 3.6}deg; background: conic-gradient({color} {score * 3.6}deg, #1e293b 0deg);">
                <span class="progress-text">{int(score)}%</span>
            </div>
            <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 0.875rem;">Match Score</p>
        </div>
    """

def analyze_resumes(job_title, job_description, required_skills, groq_api_key, files, jd_file=None):
    """Send resume analysis request to API"""
    try:
        # Prepare the request files
        files_data = []
        for file in files:
            files_data.append(('files', (file.name, file.getvalue(), file.type)))
        
        if jd_file:
            files_data.append(('jd_file', (jd_file.name, jd_file.getvalue(), jd_file.type)))

        data = {
            'job_title': job_title,
            'job_description': job_description,
            'required_skills': required_skills,
            'groq_api_key': groq_api_key
        }
        
        response = requests.post(
            f"{API_BASE}/rank-resumes",
            data=data,
            files=files_data,
            timeout=120 # Increased timeout for potential PDF processing
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def export_to_csv(results):
    """Export results to CSV"""
    data = []
    for candidate in results['top_candidates']:
        data.append({
            'Rank': candidate['rank'],
            'Filename': candidate['filename'],
            'Skill Coverage (%)': round((candidate['skill_match_count'] / candidate['total_skills_required'] * 100) if candidate['total_skills_required'] > 0 else 0),
            'Skills Matched': candidate['skill_match_count'],
            'Total Skills': candidate['total_skills_required'],
            'Matched Skills': ', '.join(candidate['matched_skills']),
            'Missing Skills': ', '.join(candidate['missing_skills']),
            'AI Verdict': candidate['ai_verdict'],
            'AI Feedback': candidate['ai_feedback']
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

def get_verdict_class(verdict):
    """Determine verdict class based on AI verdict text"""
    verdict_lower = verdict.lower()
    if any(word in verdict_lower for word in ['excellent', 'outstanding', 'perfect', 'strong fit']):
        return 'verdict-excellent'
    elif any(word in verdict_lower for word in ['good', 'suitable', 'qualified', 'potential fit']):
        return 'verdict-good'
    else:
        return 'verdict-poor'

# Sidebar
with st.sidebar:
    # Header with logo
    st.markdown("""
        <div style='display: flex; align-items: center; gap: 1rem; margin-bottom: 2rem;'>
            <div style='width: 50px; height: 50px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); 
                        border-radius: 12px; display: flex; align-items: center; justify-content: center;'>
                <span style='font-size: 24px;'>🚀</span>
            </div>
            <div>
                <h2 style='margin: 0; color: #e2e8f0;'>AI Resume Screener V3</h2>
                <p style='margin: 0; color: #94a3b8; font-size: 0.875rem;'>Agentic Candidate Ranking</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Input fields
    st.markdown("### 📝 Job Details")
    
    job_title = st.text_input(
        "Job Title",
        value="Senior Data Scientist",
        placeholder="e.g., Senior Data Scientist",
        help="Enter the job title for the position"
    )
    
    # --- NEW: Tabs for JD input ---
    tab1, tab2 = st.tabs(["Enter JD Manually", "Upload JD (PDF)"])

    job_description = ""
    jd_file = None

    with tab1:
        job_description = st.text_area(
            "Job Description",
            value="We are seeking an experienced Senior Data Scientist. Required Skills: Python, SQL, Machine Learning, TensorFlow, Scikit-learn, Pandas, AWS, Statistics.",
            placeholder="Paste the complete job description here...",
            height=150,
            help="Provide the full job description"
        )
    
    with tab2:
        jd_file = st.file_uploader(
            "Upload Job Description File",
            type=['pdf'],
            accept_multiple_files=False,
            help="Upload a single PDF file for the job description."
        )
        if jd_file:
            st.success(f"✓ JD File '{jd_file.name}' ready for analysis.")

    
    required_skills = st.text_input(
        "Required Skills (comma-separated)",
        value="Python, SQL, Machine Learning, TensorFlow, AWS",
        placeholder="Python, SQL, Machine Learning...",
        help="Separate skills with commas. These are crucial for ranking."
    )

    st.markdown("### 🔑 API Key")
    # --- FIXED: Reverted to dropdown for Groq API Key ---
    groq_api_keys = [
        "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",  # Replace with your first key
        "your-groq-api-key-2",  # Replace with your second key
        "your-groq-api-key-3"   # Replace with your third key
    ]
    
    groq_api_key = st.selectbox(
        "Groq API Key",
        options=groq_api_keys,
        help="Select a Groq API key for AI verdict & feedback generation"
    )


    st.markdown("### 📎 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose resume files",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Upload multiple resume files"
    )
    
    if uploaded_files:
        st.success(f"✓ {len(uploaded_files)} resume file(s) uploaded")
        with st.expander("View uploaded files"):
            for file in uploaded_files:
                st.markdown(f"• {file.name}")
    
    st.markdown("### ⚙️ Display Options")
    num_files = len(uploaded_files) if uploaded_files else 0
    top_n = st.number_input(
        label="Top N candidates to display",
        min_value=1,
        max_value=num_files if num_files > 0 else 1,
        value=min(5, num_files) if num_files > 0 else 1,
        step=1,
        help="Choose how many top resumes to display in the results.",
        disabled=num_files == 0
    )

    st.markdown("---")
    
    # Analyze button
    if st.button("🚀 Start Analysis", use_container_width=True, disabled=st.session_state.processing):
        is_jd_provided = bool(job_description.strip()) or (jd_file is not None)
        
        if not all([job_title, required_skills]) or not is_jd_provided:
            st.error("Please provide Job Title, Required Skills, and a Job Description (either text or PDF).")
        elif not uploaded_files:
            st.error("Please upload at least one resume")
        elif not groq_api_key or "your-groq-api-key" in groq_api_key:
            st.error("Please select a valid Groq API key from the dropdown.")
        else:
            st.session_state.processing = True
            with st.spinner("Analyzing resumes... This may take a moment."):
                # If jd_file is used, job_description will be sent as an empty string.
                # The backend logic will prioritize the file.
                current_jd_text = "" if jd_file else job_description
                
                results = analyze_resumes(
                    job_title, 
                    current_jd_text, 
                    required_skills, 
                    groq_api_key,
                    uploaded_files,
                    jd_file
                )
                if results:
                    st.session_state.analysis_results = results
                    st.success("✨ Analysis complete!")
                st.session_state.processing = False

# Main content area
st.markdown("# 📊 Analysis Results")

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    
    # Header metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Job Title", results['job_title'])
    with col2:
        st.metric("Candidates Analyzed", len(results['top_candidates']))
    with col3:
        st.metric("Processing Time", f"{results['processing_time']:.1f}s")
    with col4:
        csv = export_to_csv(results)
        st.download_button(
            label="📥 Export Results",
            data=csv,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # Display candidates based on top_n
    top_candidates_to_display = results['top_candidates'][:top_n]
    for idx, candidate in enumerate(top_candidates_to_display):
        skill_coverage = (candidate['skill_match_count'] / candidate['total_skills_required'] * 100) if candidate['total_skills_required'] > 0 else 0
        
        # Candidate header card
        st.markdown(f"""
            <div class="glass-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 style='color: #e2e8f0;'>
                        <span style='color: #94a3b8; font-size: 1.5rem;'>#{candidate['rank']}</span> 
                        &nbsp;&nbsp;{candidate['filename']}
                    </h2>
                    <div style="text-align: right;">
                        <span style="color: #94a3b8; font-size: 0.9rem;">Skills Coverage</span><br>
                        <span style="color: #e2e8f0; font-size: 1.8rem; font-weight: bold;">
                            {candidate['skill_match_count']}/{candidate['total_skills_required']}
                        </span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress and verdict row
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown(create_circular_progress(skill_coverage), unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📊 Match Details")
            st.metric("Percentage", f"{skill_coverage:.1f}%")
            if skill_coverage >= 80:
                st.success("Excellent Match")
            elif skill_coverage >= 60:
                st.warning("Good Match")
            else:
                st.error("Needs Review")
        
        with col3:
            st.markdown("### 🤖 AI Verdict")
            verdict_class = get_verdict_class(candidate['ai_verdict'])
            st.markdown(f"""
                <div class="feedback-box">
                    <span class="verdict-badge {verdict_class}">{candidate['ai_verdict']}</span>
                    <p style="color: #cbd5e1; margin-top: 0.5rem; line-height: 1.6;">
                        {candidate['ai_feedback']}
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Evidence and skills sections
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📝 Key Evidence")
            # Format the text to ensure it displays as a continuous paragraph
            evidence_text = candidate['key_matching_chunk'].replace('\n\n', ' ').replace('\n', ' ').strip()
            st.markdown(f"""
                <div class="evidence-box">
                    {evidence_text}
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Skills Analysis")
            
            # Matched skills
            st.markdown(f"**✅ Matched Skills ({len(candidate['matched_skills'])})**")
            if candidate['matched_skills']:
                skills_html = "".join([f"<span class='skill-matched'>{skill}</span>" for skill in candidate['matched_skills']])
                st.markdown(f"<div style='margin-bottom: 1rem;'>{skills_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #64748b; font-style: italic;'>No matched skills</p>", unsafe_allow_html=True)
            
            # Missing skills
            st.markdown(f"**❌ Missing Skills ({len(candidate['missing_skills'])})**")
            if candidate['missing_skills']:
                skills_html = "".join([f"<span class='skill-missing'>{skill}</span>" for skill in candidate['missing_skills']])
                st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<p style='color: #64748b; font-style: italic;'>No missing skills</p>", unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
        <div style='text-align: center; padding: 4rem 0;'>
            <div style='font-size: 4rem; margin-bottom: 1rem;'>📄</div>
            <h2 style='color: #e2e8f0;'>Ready to Analyze</h2>
            <p style='color: #94a3b8; max-width: 500px; margin: 0 auto;'>
                Fill in the details on the left, upload resumes, and click "Start Analysis" to begin intelligent candidate ranking.
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.875rem; padding: 2rem 0;'>
        <p>AI Resume Screener V3 • Powered by Advanced NLP & Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

