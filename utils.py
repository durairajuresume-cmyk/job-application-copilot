import io
import json
import re
from collections import Counter

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
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    pages = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n\n".join(pages).strip()


def analyze_pdf_format(uploaded_file) -> dict:
    """Extract font-size and accent-color hints from the first page of the original PDF."""
    fmt: dict = {
        "name_size": 20,
        "heading_size": 11,
        "body_size": 10,
        "contact_size": 9,
        "heading_rgb": None,  # None → fall back to dark gray
    }
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        with pdfplumber.open(uploaded_file) as pdf:
            if not pdf.pages:
                return fmt
            chars = [c for c in pdf.pages[0].chars if c.get("size", 0) > 0]
            if not chars:
                return fmt

            # ── Font sizes ────────────────────────────────────────────────
            sizes = [round(float(c["size"])) for c in chars]
            cnt = Counter(sizes)
            body = cnt.most_common(1)[0][0]
            fmt["body_size"] = max(body, 8)
            fmt["contact_size"] = max(body - 1, 7)
            larger = sorted(s for s in cnt if s > body)
            fmt["heading_size"] = larger[0] if larger else body + 2
            fmt["name_size"] = max(sizes)
            # Ensure sane hierarchy
            if fmt["name_size"] <= fmt["heading_size"]:
                fmt["name_size"] = fmt["heading_size"] + 4
            if fmt["heading_size"] <= fmt["body_size"]:
                fmt["heading_size"] = fmt["body_size"] + 1

            # ── Accent color ──────────────────────────────────────────────
            for char in chars:
                raw = char.get("non_stroking_color")
                if raw is None or isinstance(raw, (int, float)):
                    continue  # grayscale — not an accent
                if isinstance(raw, (list, tuple)) and len(raw) == 3:
                    r, g, b = float(raw[0]), float(raw[1]), float(raw[2])
                    # pdfplumber returns 0-1 floats
                    if max(r, g, b) <= 1.0:
                        r, g, b = r * 255, g * 255, b * 255
                    ri, gi, bi = round(r), round(g), round(b)
                    # Skip near-black (< 60 each) and near-white (> 195 each)
                    is_black = ri < 60 and gi < 60 and bi < 60
                    is_white = ri > 195 and gi > 195 and bi > 195
                    if not is_black and not is_white:
                        fmt["heading_rgb"] = (ri, gi, bi)
                        break
    except Exception:
        pass
    return fmt


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


def _safe(text: str) -> str:
    """Map common Unicode chars to Latin-1 equivalents for fpdf2 core fonts."""
    if not text:
        return ""
    text = text.translate(str.maketrans({
        '–': '-',    # en dash
        '—': '-',    # em dash
        '‘': "'",    # left single quote
        '’': "'",    # right single quote / apostrophe
        '“': '"',    # left double quote
        '”': '"',    # right double quote
        '•': '-',    # bullet
        '…': '...',  # ellipsis
        ' ': ' ',    # non-breaking space
        '­': '',     # soft hyphen
    }))
    return text.encode('latin-1', errors='replace').decode('latin-1')


def generate_resume_pdf(data: dict, fmt: dict | None = None) -> bytes:
    """Render structured resume data matching the style hints in fmt."""

    if fmt is None:
        fmt = {}

    # Clamp sizes to sane ranges
    name_sz = max(14, min(int(fmt.get("name_size", 20)), 30))
    head_sz = max(9, min(int(fmt.get("heading_size", 11)), 18))
    body_sz = max(8, min(int(fmt.get("body_size", 10)), 14))
    cont_sz = max(7, min(int(fmt.get("contact_size", 9)), 12))
    head_rgb = fmt.get("heading_rgb") or (30, 30, 30)

    class _PDF(FPDF):
        def header(self):
            pass
        def footer(self):
            pass

    def _section_heading(pdf: FPDF, title: str):
        pdf.set_font("Helvetica", "B", head_sz)
        pdf.set_text_color(*head_rgb)
        pdf.cell(0, 7, _safe(title.upper()), ln=True)
        pdf.set_draw_color(*head_rgb)
        pdf.set_line_width(0.3)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 170, pdf.get_y())
        pdf.set_text_color(40, 40, 40)
        pdf.ln(2)

    pdf = _PDF()
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # Name
    pdf.set_font("Helvetica", "B", name_sz)
    pdf.set_text_color(15, 15, 15)
    pdf.cell(0, 10, _safe(data.get("name", "")), ln=True, align="C")

    # Contact
    pdf.set_font("Helvetica", "", cont_sz)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, _safe(data.get("contact", "")), ln=True, align="C")
    pdf.ln(5)

    # Summary
    if data.get("summary"):
        _section_heading(pdf, "Professional Summary")
        pdf.set_font("Helvetica", "", body_sz)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, _safe(data["summary"]))
        pdf.ln(4)

    # Experience
    if data.get("experience"):
        _section_heading(pdf, "Experience")
        for exp in data["experience"]:
            pdf.set_font("Helvetica", "B", body_sz)
            pdf.set_text_color(20, 20, 20)
            title_line = exp.get("title", "")
            company = exp.get("company", "")
            if company:
                title_line += f"  -  {company}"
            pdf.cell(0, 6, _safe(title_line), ln=True)

            pdf.set_font("Helvetica", "I", cont_sz)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 4, _safe(exp.get("duration", "")), ln=True)

            pdf.set_font("Helvetica", "", body_sz)
            pdf.set_text_color(40, 40, 40)
            for bullet in exp.get("bullets", []):
                pdf.cell(5, 5, "-", ln=False)
                pdf.multi_cell(0, 5, _safe(bullet))
            pdf.ln(3)

    # Skills
    if data.get("skills"):
        _section_heading(pdf, "Skills")
        pdf.set_font("Helvetica", "", body_sz)
        pdf.set_text_color(40, 40, 40)
        pdf.multi_cell(0, 5, "  -  ".join(_safe(s) for s in data["skills"]))
        pdf.ln(4)

    # Education
    if data.get("education"):
        _section_heading(pdf, "Education")
        for edu in data["education"]:
            pdf.set_font("Helvetica", "B", body_sz)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(0, 6, _safe(edu.get("degree", "")), ln=True)
            pdf.set_font("Helvetica", "", body_sz)
            pdf.set_text_color(60, 60, 60)
            inst = _safe(edu.get("institution", ""))
            year = _safe(edu.get("year", ""))
            pdf.cell(0, 5, f"{inst}  {year}".strip(), ln=True)
            pdf.ln(2)

    return bytes(pdf.output())
