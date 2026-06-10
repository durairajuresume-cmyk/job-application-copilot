import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from utils import (
    analyze_pdf_format,
    analyze_resume,
    extract_text_from_pdf,
    generate_cover_letter,
    generate_interview_prep,
    generate_resume_pdf,
    rewrite_resume,
)

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Durai's Job Application Copilot",
    page_icon="assets/icon.png" if os.path.exists("assets/icon.png") else "💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API key resolution: st.secrets → .env → error ───────────────────────────
load_dotenv()
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    st.error(
        "ANTHROPIC_API_KEY not found. "
        "Add it to `.streamlit/secrets.toml` or a `.env` file and restart the app."
    )
    st.stop()

client = anthropic.Anthropic(api_key=api_key)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ---- global ---- */
[data-testid="stAppViewContainer"] { background: #f8fafc; }
[data-testid="stSidebar"] { background: #1e293b; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

/* ---- header ---- */
.app-header { text-align: center; padding: 1.5rem 0 0.5rem; }
.app-header h1 { font-size: 2.4rem; font-weight: 800; color: #1e293b; margin: 0; }
.app-header p  { color: #64748b; font-size: 1.1rem; margin-top: 0.4rem; }

/* ---- score card ---- */
.score-card {
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: center;
    color: white;
    margin-bottom: 1.5rem;
}
.score-label { font-size: 0.9rem; font-weight: 600; letter-spacing: 0.08em; opacity: 0.85; }
.score-number { font-size: 5rem; font-weight: 900; line-height: 1; margin: 0.25rem 0; }
.score-verdict { font-size: 1.1rem; font-weight: 600; }
.score-rationale { font-size: 0.85rem; opacity: 0.85; margin-top: 0.5rem; }

/* ---- list items ---- */
.item-green  { background:#f0fdf4; border-left:4px solid #22c55e; padding:.65rem 1rem; margin:.4rem 0; border-radius:0 8px 8px 0; color:#166534; font-size:.95rem; }
.item-orange { background:#fff7ed; border-left:4px solid #f97316; padding:.65rem 1rem; margin:.4rem 0; border-radius:0 8px 8px 0; color:#9a3412; font-size:.95rem; }
.item-blue   { background:#eff6ff; border-left:4px solid #3b82f6; padding:.65rem 1rem; margin:.4rem 0; border-radius:0 8px 8px 0; color:#1e40af; font-size:.95rem; }

/* ---- section headings ---- */
.section-title { font-size:1.1rem; font-weight:700; color:#1e293b; margin:1.2rem 0 0.6rem; }

/* ---- tabs ---- */
[data-baseweb="tab"] { font-size:1rem; font-weight:600; }

/* ---- primary button ---- */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border: none;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.75rem 2rem;
    border-radius: 10px;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border: none;
}

/* ---- divider ---- */
hr { border-color: #e2e8f0; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## How it works")
    st.markdown(
        """
1. Upload your resume (PDF)
2. Paste the job description
3. Click **Analyze**

The app calls Claude to produce:
- A **match score** with rationale
- **Strengths**, **gaps**, and **resume improvements**
- A **tailored cover letter**
- **Interview prep** with STAR answers
- A **rewritten resume** tailored to the role (PDF download)
"""
    )
    st.markdown("---")
    st.caption("Powered by Claude · Built with Streamlit")

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="app-header">
  <h1>Durai's Job Application Copilot</h1>
  <p>AI-powered resume analysis, cover letters, and interview coaching</p>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── Input section ────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### Resume (PDF)")
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.success(f"Uploaded: **{uploaded_file.name}**")

with col_right:
    st.markdown("### Job Description")
    job_description = st.text_area(
        "Paste the job description",
        height=260,
        placeholder="Paste the full job posting here — include responsibilities, requirements, and nice-to-haves for the best analysis.",
        label_visibility="collapsed",
    )

st.markdown("")
analyze_clicked = st.button(
    "Analyze My Application",
    type="primary",
    use_container_width=True,
    disabled=(not uploaded_file or not job_description.strip()),
)

# ── Analysis logic ───────────────────────────────────────────────────────────
if analyze_clicked:
    status = st.status("Running analysis…", expanded=True)
    try:
        status.write("Extracting text from PDF…")
        resume_text = extract_text_from_pdf(uploaded_file)
        pdf_fmt = analyze_pdf_format(uploaded_file)  # seek(0) is inside the function
        if not resume_text:
            status.update(label="Extraction failed", state="error")
            st.error(
                "Could not extract text from this PDF. "
                "Please ensure it is not a scanned image or a locked document."
            )
            st.stop()

        status.write("Analyzing resume against the job description…")
        analysis = analyze_resume(client, resume_text, job_description)

        status.write("Writing tailored cover letter…")
        cover_letter = generate_cover_letter(client, resume_text, job_description)

        status.write("Building interview preparation guide…")
        interview_prep = generate_interview_prep(client, resume_text, job_description)

        status.write("Rewriting resume for this role…")
        rewritten = rewrite_resume(client, resume_text, job_description)

        status.update(label="Analysis complete!", state="complete", expanded=False)

        st.session_state["analysis"] = analysis
        st.session_state["cover_letter"] = cover_letter
        st.session_state["interview_prep"] = interview_prep
        st.session_state["rewritten"] = rewritten
        st.session_state["pdf_fmt"] = pdf_fmt
        st.session_state["has_results"] = True

    except anthropic.AuthenticationError:
        status.update(label="Authentication failed", state="error")
        st.error("Invalid API key. Please check your Anthropic API key and try again.")
    except Exception as exc:
        status.update(label="Error", state="error")
        st.error(f"Something went wrong: {exc}")

# ── Results ──────────────────────────────────────────────────────────────────
if st.session_state.get("has_results"):
    analysis: dict = st.session_state["analysis"]
    cover_letter: str = st.session_state["cover_letter"]
    interview_prep: str = st.session_state["interview_prep"]
    rewritten: dict = st.session_state["rewritten"]

    score: int = int(analysis.get("match_score", 0))
    rationale: str = analysis.get("score_rationale", "")
    strengths: list = analysis.get("strengths", [])
    missing: list = analysis.get("missing_skills", [])
    improvements: list = analysis.get("improvements", [])

    # Score card colour
    if score >= 71:
        bg = "linear-gradient(135deg, #16a34a, #15803d)"
        verdict = "Strong Match"
    elif score >= 41:
        bg = "linear-gradient(135deg, #ea580c, #c2410c)"
        verdict = "Moderate Match"
    else:
        bg = "linear-gradient(135deg, #dc2626, #b91c1c)"
        verdict = "Low Match"

    st.markdown("---")
    st.markdown("## Results")

    # Score card (centred, 1/3 width)
    _, score_col, _ = st.columns([1, 2, 1])
    with score_col:
        st.markdown(
            f"""
<div class="score-card" style="background:{bg}">
  <div class="score-label">MATCH SCORE</div>
  <div class="score-number">{score}</div>
  <div class="score-verdict">{verdict}</div>
  <div class="score-rationale">{rationale}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Tabs
    tab_analysis, tab_cover, tab_interview, tab_rewrite = st.tabs(
        ["Resume Analysis", "Cover Letter", "Interview Prep", "Rewritten Resume"]
    )

    # ── Tab 1: Analysis ──────────────────────────────────────────────────
    with tab_analysis:
        col_a, col_b = st.columns(2, gap="large")

        with col_a:
            st.markdown('<div class="section-title">Strengths</div>', unsafe_allow_html=True)
            for item in strengths:
                st.markdown(f'<div class="item-green">✓ {item}</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">Missing Skills / Gaps</div>', unsafe_allow_html=True)
            for item in missing:
                st.markdown(f'<div class="item-orange">✗ {item}</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-title">Resume Improvement Suggestions</div>', unsafe_allow_html=True)
            for i, item in enumerate(improvements, 1):
                st.markdown(f'<div class="item-blue">{i}. {item}</div>', unsafe_allow_html=True)

    # ── Tab 2: Cover Letter ──────────────────────────────────────────────
    with tab_cover:
        st.markdown("#### Tailored Cover Letter")
        st.markdown(
            "Review and personalise before sending — replace **[Company Name]** with the actual company.",
            help="This letter was generated from your resume and the job description.",
        )
        st.text_area(
            "Cover Letter",
            value=cover_letter,
            height=420,
            label_visibility="collapsed",
        )
        st.download_button(
            label="Download as .txt",
            data=cover_letter,
            file_name="cover_letter.txt",
            mime="text/plain",
        )

    # ── Tab 3: Interview Prep ────────────────────────────────────────────
    with tab_interview:
        st.markdown("#### Interview Preparation Guide")
        st.markdown(interview_prep)

    # ── Tab 4: Rewritten Resume ──────────────────────────────────────────
    with tab_rewrite:
        st.markdown("#### Rewritten Resume")
        st.markdown(
            "Tailored to this job description. Review carefully — download when ready.",
            help="Claude rewrote your resume using only your existing experience and skills.",
        )

        # Changes summary
        changes = rewritten.get("changes", [])
        if changes:
            with st.expander("What was changed", expanded=True):
                for change in changes:
                    st.markdown(f"- {change}")

        st.markdown("---")

        # Rendered resume preview
        col_r, col_info = st.columns([2, 1], gap="large")

        with col_r:
            st.markdown(f"## {rewritten.get('name', '')}")
            st.caption(rewritten.get("contact", ""))

            if rewritten.get("summary"):
                st.markdown("**Professional Summary**")
                st.markdown(rewritten["summary"])

            if rewritten.get("experience"):
                st.markdown("**Experience**")
                for exp in rewritten["experience"]:
                    st.markdown(
                        f"**{exp.get('title', '')}** — {exp.get('company', '')}  \n"
                        f"*{exp.get('duration', '')}*"
                    )
                    for bullet in exp.get("bullets", []):
                        st.markdown(f"- {bullet}")

            if rewritten.get("skills"):
                st.markdown("**Skills**")
                st.markdown(" · ".join(rewritten["skills"]))

            if rewritten.get("education"):
                st.markdown("**Education**")
                for edu in rewritten["education"]:
                    st.markdown(
                        f"**{edu.get('degree', '')}** — "
                        f"{edu.get('institution', '')} {edu.get('year', '')}"
                    )

        with col_info:
            st.markdown("**Download**")
            pdf_bytes = generate_resume_pdf(rewritten, st.session_state.get("pdf_fmt"))
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="rewritten_resume.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            txt = "\n\n".join([
                rewritten.get("name", ""),
                rewritten.get("contact", ""),
                "SUMMARY\n" + rewritten.get("summary", ""),
                "EXPERIENCE\n" + "\n".join(
                    f"{e.get('title')} — {e.get('company')} ({e.get('duration')})\n" +
                    "\n".join(f"• {b}" for b in e.get("bullets", []))
                    for e in rewritten.get("experience", [])
                ),
                "SKILLS\n" + " • ".join(rewritten.get("skills", [])),
                "EDUCATION\n" + "\n".join(
                    f"{e.get('degree')} — {e.get('institution')} {e.get('year', '')}"
                    for e in rewritten.get("education", [])
                ),
            ])
            st.download_button(
                label="Download TXT",
                data=txt,
                file_name="rewritten_resume.txt",
                mime="text/plain",
                use_container_width=True,
            )
