# Job Application Copilot

An AI-powered Streamlit app that analyzes your resume against a job description and produces:

- A **match score** (0–100) with rationale
- **Strengths** that align with the role
- **Missing skills** and qualification gaps
- **Resume improvement suggestions** tailored to the posting
- A **tailored cover letter** ready to customize
- A complete **interview preparation guide** with STAR-format answers

Built with Python, Streamlit, and Claude (Anthropic).

---

## Prerequisites

- Python 3.10 or later
- An [Anthropic API key](https://console.anthropic.com/)

---

## Quickstart

```bash
# 1. Clone or enter the project directory
cd job-application-copilot

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Usage

1. Enter your **Anthropic API key** in the sidebar.
2. Upload your **resume as a PDF** (text-based, not scanned).
3. Paste the **full job description** into the text area.
4. Click **Analyze My Application**.

Results appear in three tabs:

| Tab | Content |
|-----|---------|
| Resume Analysis | Match score, strengths, gaps, improvement suggestions |
| Cover Letter | Tailored letter ready to copy or download |
| Interview Prep | Questions, STAR answers, talking points, questions to ask |

---

## Project Structure

```
job-application-copilot/
├── app.py           # Streamlit UI and application logic
├── prompts.py       # All Claude prompt templates
├── utils.py         # PDF extraction and Claude API helpers
├── requirements.txt
└── README.md
```

---

## Notes

- **Scanned PDFs** (image-only) cannot be parsed. Use a text-based PDF or run OCR first.
- Your API key is used only within your local session and is never stored.
- Results quality improves with detailed job descriptions (include responsibilities and requirements sections).
