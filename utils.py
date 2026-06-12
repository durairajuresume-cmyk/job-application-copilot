import html as _html
import io
import json
import re
from collections import Counter

import anthropic
import pdfplumber

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
        "heading_rgb": None,  # None -> fall back to dark gray
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

            # Font sizes
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

            # Accent color
            for char in chars:
                raw = char.get("non_stroking_color")
                if raw is None or isinstance(raw, (int, float)):
                    continue  # grayscale -- not an accent
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


def fetch_job_from_url(client: anthropic.Anthropic, url: str) -> dict:
    """Fetch a job posting URL, strip HTML, use Claude to extract title/company/description."""
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
        tag.decompose()
    raw_text = soup.get_text(separator="\n", strip=True)[:8000]

    prompt = (
        "Extract the job posting from this webpage text and return a JSON object with "
        "exactly four fields:\n"
        '- "title": the job title\n'
        '- "company": the company name\n'
        '- "location": the job location (city, state, country, or "Remote" — empty string if not found)\n'
        '- "description": the full job description (responsibilities, requirements, '
        "qualifications — keep it complete)\n\n"
        f"<webpage_text>\n{raw_text}\n</webpage_text>\n\n"
        "Return only the JSON object, no markdown, no extra text."
    )
    raw = _call_claude(
        client,
        "You extract structured job posting data from webpage text. Return only valid JSON.",
        prompt,
        max_tokens=2000,
    )
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError("Could not parse job data from this page.")
    return json.loads(json_match.group())


def rank_job_matches(client: anthropic.Anthropic, resume_text: str, postings: list) -> str:
    """Ask Claude to rank retrieved job postings by fit and explain each."""
    postings_block = "\n\n".join(
        f"[{i + 1}] {p.get('title', 'Untitled')} at {p.get('company', 'Unknown')}"
        f" (similarity: {p.get('similarity', 0):.2f})\n"
        f"{p.get('description', '')[:500]}..."
        for i, p in enumerate(postings)
    )
    prompt = (
        f"Here is the candidate's resume (first 2000 characters):\n"
        f"<resume>\n{resume_text[:2000]}\n</resume>\n\n"
        f"These {len(postings)} job postings were retrieved by semantic similarity "
        f"to the resume:\n\n{postings_block}\n\n"
        f"Rank them from best to worst fit. For each posting provide:\n"
        f"1. Rank and job title\n"
        f"2. Why it is a strong or weak fit (2 sentences)\n"
        f"3. One concrete tip to strengthen the application\n\n"
        f"Return a clean numbered list."
    )
    return _call_claude(client, "You are an expert career coach.", prompt, max_tokens=2000)


def _safe(text: str) -> str:
    """Normalize common Unicode punctuation so text is safe for ReportLab core fonts."""
    if not text:
        return ""
    # Replace smart quotes, dashes, bullets, etc. with plain ASCII equivalents
    replacements = {
        0x2013: "-",   # en dash
        0x2014: "-",   # em dash
        0x2018: "'",   # left single quote
        0x2019: "'",   # right single quote / apostrophe
        0x201C: '"',   # left double quote
        0x201D: '"',   # right double quote
        0x2022: "-",   # bullet
        0x2026: "...", # ellipsis
        0x00A0: " ",   # non-breaking space
        0x00AD: "",    # soft hyphen
    }
    return str(text).translate(replacements)


def generate_resume_pdf(data: dict, fmt: dict | None = None) -> bytes:
    """Render structured resume JSON into a style-faithful PDF using ReportLab.

    ReportLab's Paragraph/Platypus flowable model handles text reflow, true
    hanging indents, and automatic page breaks -- no manual cursor arithmetic.
    The extracted format hints (font sizes, accent color) from the user's
    original PDF are applied via ParagraphStyle so any uploaded resume style
    is faithfully reflected in the output.
    """
    from reportlab.lib.colors import Color
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

    if fmt is None:
        fmt = {}

    # Clamp extracted sizes to sane ranges
    name_sz = max(14, min(int(fmt.get("name_size",    20)), 30))
    head_sz = max( 9, min(int(fmt.get("heading_size", 11)), 18))
    body_sz = max( 8, min(int(fmt.get("body_size",    10)), 14))
    cont_sz = max( 7, min(int(fmt.get("contact_size",  9)), 12))
    r, g, b = fmt.get("heading_rgb") or (30, 30, 30)

    accent   = Color(r / 255, g / 255, b / 255)
    dark     = Color(0.08, 0.08, 0.08)
    body_clr = Color(0.25, 0.25, 0.25)
    muted    = Color(0.40, 0.40, 0.40)

    CONTENT_W   = 170 * mm
    BULLET_HANG = 5   * mm  # dash overhangs; wrapped lines align with text start

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    name_sty = ps("nm",
        fontName="Helvetica-Bold", fontSize=name_sz, leading=name_sz * 1.2,
        textColor=dark, alignment=TA_CENTER, spaceAfter=1 * mm)
    contact_sty = ps("ct",
        fontName="Helvetica", fontSize=cont_sz, leading=cont_sz * 1.4,
        textColor=muted, alignment=TA_CENTER, spaceAfter=4 * mm)
    sec_sty = ps("sec",
        fontName="Helvetica-Bold", fontSize=head_sz, leading=head_sz * 1.4,
        textColor=accent, alignment=TA_LEFT, spaceBefore=3 * mm, spaceAfter=1 * mm)
    job_title_sty = ps("jt",
        fontName="Helvetica-Bold", fontSize=body_sz, leading=body_sz * 1.4,
        textColor=dark, alignment=TA_LEFT, spaceBefore=2 * mm, spaceAfter=0.5 * mm)
    duration_sty = ps("du",
        fontName="Helvetica-Oblique", fontSize=cont_sz, leading=cont_sz * 1.4,
        textColor=muted, alignment=TA_LEFT, spaceAfter=1 * mm)
    body_sty = ps("bd",
        fontName="Helvetica", fontSize=body_sz, leading=body_sz * 1.4,
        textColor=body_clr, alignment=TA_JUSTIFY, spaceAfter=3 * mm)
    # True hanging indent: "- " prefix sits at the left edge; continuation
    # lines wrap flush with the text start (firstLineIndent pulls first line left).
    bullet_sty = ps("bu",
        fontName="Helvetica", fontSize=body_sz, leading=body_sz * 1.4,
        textColor=body_clr, alignment=TA_JUSTIFY,
        leftIndent=BULLET_HANG, firstLineIndent=-BULLET_HANG,
        spaceAfter=1 * mm)
    skills_sty = ps("sk",
        fontName="Helvetica", fontSize=body_sz, leading=body_sz * 1.5,
        textColor=body_clr, alignment=TA_LEFT, spaceAfter=3 * mm)
    edu_deg_sty = ps("ed",
        fontName="Helvetica-Bold", fontSize=body_sz, leading=body_sz * 1.4,
        textColor=dark, spaceBefore=1 * mm, spaceAfter=0.5 * mm)
    edu_info_sty = ps("ei",
        fontName="Helvetica", fontSize=body_sz, leading=body_sz * 1.4,
        textColor=muted, spaceAfter=2 * mm)

    story = []

    def section(title):
        story.append(Paragraph(title, sec_sty))
        story.append(HRFlowable(
            width=CONTENT_W, thickness=0.4, color=accent, spaceAfter=2 * mm))

    def t(text):
        """Normalize + HTML-escape text so Paragraph markup won't misfire on user content."""
        return _html.escape(_safe(str(text) if text else ""))

    def add_bullet(text):
        escaped = t(text)
        if escaped:
            story.append(Paragraph(f"- {escaped}", bullet_sty))

    # Name & contact
    story.append(Paragraph(t(data.get("name", "")), name_sty))
    if data.get("contact"):
        story.append(Paragraph(t(data["contact"]), contact_sty))

    # Summary
    if data.get("summary"):
        section("PROFESSIONAL SUMMARY")
        story.append(Paragraph(t(data["summary"]), body_sty))

    # Experience
    if data.get("experience"):
        section("EXPERIENCE")
        for exp in data["experience"]:
            title   = t(exp.get("title",   ""))
            company = t(exp.get("company", ""))
            header  = f"{title}  -  {company}" if company else title
            story.append(Paragraph(header, job_title_sty))
            if exp.get("duration"):
                story.append(Paragraph(t(exp["duration"]), duration_sty))
            for b in exp.get("bullets", []):
                add_bullet(b.strip())
            story.append(Spacer(1, 2 * mm))

    # Skills
    if data.get("skills"):
        section("SKILLS")
        skills_line = "  |  ".join(t(sk) for sk in data["skills"] if sk)
        story.append(Paragraph(skills_line, skills_sty))

    # Education
    if data.get("education"):
        section("EDUCATION")
        for edu in data["education"]:
            story.append(Paragraph(t(edu.get("degree", "")), edu_deg_sty))
            inst = t(edu.get("institution", ""))
            year = t(edu.get("year", ""))
            if inst or year:
                story.append(Paragraph(f"{inst}  {year}".strip(), edu_info_sty))

    # Projects
    if data.get("projects"):
        section("PROJECTS")
        for proj in data["projects"]:
            title  = t(proj.get("title", ""))
            tech   = t(proj.get("tech",  ""))
            header = f"{title}  -  {tech}" if tech else title
            story.append(Paragraph(header, job_title_sty))
            for b in proj.get("bullets", []):
                add_bullet(b.strip())
            story.append(Spacer(1, 2 * mm))

    # Certifications
    if data.get("certifications"):
        section("CERTIFICATIONS")
        for cert in data["certifications"]:
            add_bullet(str(cert).strip())

    # Awards
    if data.get("awards"):
        section("AWARDS & RECOGNITION")
        for award in data["awards"]:
            add_bullet(str(award).strip())

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
    )
    doc.build(story)
    return buf.getvalue()
