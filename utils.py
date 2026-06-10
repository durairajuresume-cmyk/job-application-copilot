import io
import json
import re

import anthropic
import pdfplumber
from fpdf import FPDF

from prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_PROMPT,
    COVER_LETTER_SYSTEM_PROMPT,
    COVER_LETTER_USER_PROMPT,
    INTERVIEW_PREP_SYSTEM_PROMPT,
    INTERVIEW_PREP_USER_PROMPT,
    REWRITER_SYSTEM_PROMPT,
    REWRITER_USER_PROMPT,
)

MODEL = "claude-sonnet-4-6"


def extract_text_from_pdf(uploaded_file) -> str:
    """Return concatenated text from all pages of a PDF file object."""
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages).strip()


def _call_claude(client: anthropic.Anthropic, system: str, user: str, max_tokens: int = 2048) -> str:
    message = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def analyze_resume(client: anthropic.Anthropic, resume_text: str, job_description: str) -> dict:
    """Return structured analysis dict with match_score, strengths, missing_skills, improvements."""
    prompt = ANALYSIS_USER_PROMPT.format(
        resume_text=resume_text,
        job_description=job_description,
    )
    raw = _call_claude(client, ANALYSIS_SYSTEM_PROMPT, prompt, max_tokens=1500)

    # Strip any accidental markdown code fences before parsing
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in analysis response:\n{raw}")
    return json.loads(json_match.group())


def generate_cover_letter(client: anthropic.Anthropic, resume_text: str, job_description: str) -> str:
    """Return a tailored cover letter as plain text."""
    prompt = COVER_LETTER_USER_PROMPT.format(
        resume_text=resume_text,
        job_description=job_description,
    )
    return _call_claude(client, COVER_LETTER_SYSTEM_PROMPT, prompt, max_tokens=1024)


def generate_interview_prep(client: anthropic.Anthropic, resume_text: str, job_description: str) -> str:
    """Return formatted interview preparation tips as markdown text."""
    prompt = INTERVIEW_PREP_USER_PROMPT.format(
        resume_text=resume_text,
        job_description=job_description,
    )
    return _call_claude(client, INTERVIEW_PREP_SYSTEM_PROMPT, prompt, max_tokens=2048)


def rewrite_resume(client: anthropic.Anthropic, resume_text: str, job_description: str) -> dict:
    """Return a structured rewritten resume dict tailored to the job description."""
    prompt = REWRITER_USER_PROMPT.format(
        resume_text=resume_text,
        job_description=job_description,
    )
    raw = _call_claude(client, REWRITER_SYSTEM_PROMPT, prompt, max_tokens=3000)
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON found in rewriter response:\n{raw}")
    return json.loads(json_match.group())


def generate_resume_pdf(data: dict) -> bytes:
    """Render structured resume data as a clean ATS-friendly PDF and return bytes."""

    class _PDF(FPDF):
        def header(self):
            pass
        def footer(self):
            pass

    def _section_heading(pdf: FPDF, title: str):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 7, title.upper(), ln=True)
        pdf.set_draw_color(180, 180, 180)
        pdf.set_line_width(0.3)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
        pdf.ln(2)

    pdf = _PDF()
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # Name
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(15, 15, 15)
    pdf.cell(0, 10, data.get("name", ""), ln=True, align="C")

    # Contact
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, data.get("contact", ""), ln=True, align="C")
    pdf.ln(5)

    # Summary
    if data.get("summary"):
        _section_heading(pdf, "Professional Summary")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, data["summary"])
        pdf.ln(4)

    # Experience
    if data.get("experience"):
        _section_heading(pdf, "Experience")
        for exp in data["experience"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(20, 20, 20)
            title_line = exp.get("title", "")
            company = exp.get("company", "")
            if company:
                title_line += f"  —  {company}"
            pdf.cell(0, 6, title_line, ln=True)

            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, exp.get("duration", ""), ln=True)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            for bullet in exp.get("bullets", []):
                pdf.cell(5, 5, "•", ln=False)
                pdf.multi_cell(0, 5, bullet)
            pdf.ln(3)

    # Skills
    if data.get("skills"):
        _section_heading(pdf, "Skills")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, "  •  ".join(data["skills"]))
        pdf.ln(4)

    # Education
    if data.get("education"):
        _section_heading(pdf, "Education")
        for edu in data["education"]:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(0, 6, edu.get("degree", ""), ln=True)
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            inst = edu.get("institution", "")
            year = edu.get("year", "")
            pdf.cell(0, 5, f"{inst}  {year}".strip(), ln=True)
            pdf.ln(2)

    return bytes(pdf.output())
