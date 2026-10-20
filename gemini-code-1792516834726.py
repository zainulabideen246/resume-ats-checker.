import os
import streamlit as st
import pdfplumber
from google import genai
from google.genai import types

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Resume ATS Analyzer",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------------
# GEMINI CLIENT SETUP
# ---------------------------------------------------------
# Retrieve API Key from Environment variable or Streamlit Secrets
api_key = os.getenv("GEMINI_API_KEY")

if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    st.sidebar.error("⚠️ `GEMINI_API_KEY` not found! Please set it in environment variables or Streamlit secrets.")
    client = None
else:
    client = genai.Client(api_key=api_key)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extracts raw text from an uploaded PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def analyze_resume(resume_text: str, job_description: str) -> str:
    """Queries Gemini 1.5 Flash to evaluate the resume against ATS metrics."""
    prompt = f"""
You are an expert Applicant Tracking System (ATS) optimization specialist and hiring manager.
Analyze the following candidate resume against the provided Job Description.

--- JOB DESCRIPTION ---
{job_description}

--- CANDIDATE RESUME ---
{resume_text}

Provide a structured analysis with the following exact headings and details:

1. **ATS Match Score**: Give an overall compatibility score from 0% to 100%. Explain the rating briefly.
2. **Key Strengths**: Highlight candidate skills, keywords, and experiences that align well with the job requirements.
3. **Missing Keywords & Skill Gaps**: List key technical or soft skills, tools, and certifications mentioned in the job description that are missing or weak in the resume.
4. **Actionable Suggestions for Improvement**: Provide clear, bulleted recommendations to rewrite sections or bullet points to improve the ATS parseability and match score.
"""

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
        )
    )
    return response.text

# ---------------------------------------------------------
# MAIN APPLICATION UI
# ---------------------------------------------------------
def main():
    st.title("📄 AI Resume ATS Score & Optimization Checker")
    st.markdown("Upload your resume and paste a job description to get instant ATS feedback and improvements.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Upload Resume")
        uploaded_file = st.file_uploader("Upload PDF format", type=["pdf"])
        
        resume_text = ""
        if uploaded_file is not None:
            resume_text = extract_text_from_pdf(uploaded_file)
            if resume_text.strip():
                st.success("✅ Resume parsed successfully!")
                with st.expander("Preview Extracted Text"):
                    st.text_area("Extracted Resume Content", resume_text, height=200)
            else:
                st.warning("Could not extract readable text from this PDF. Please ensure it is not a scanned image.")

    with col2:
        st.subheader("2. Job Description")
        job_description = st.text_area(
            "Paste the Target Job Description here...",
            height=250,
            placeholder="Paste duties, technical requirements, and qualifications..."
        )

    st.markdown("---")

    if st.button("🚀 Analyze Resume with Gemini AI", type="primary", use_container_width=True):
        if not client:
            st.error("Cannot proceed without a valid `GEMINI_API_KEY`.")
            return

        if not resume_text.strip():
            st.error("Please upload a valid PDF resume first.")
            return

        if not job_description.strip():
            st.error("Please paste a job description to score against.")
            return

        with st.spinner("Analyzing resume against job parameters with Gemini 1.5 Flash..."):
            try:
                feedback = analyze_resume(resume_text, job_description)
                st.subheader("📊 ATS Evaluation & Improvement Report")
                st.markdown(feedback)
            except Exception as err:
                st.error(f"Failed to analyze resume: {err}")

if __name__ == "__main__":
    main()