import json
import re

import anthropic
import pdfplumber

from prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    ANALYSIS_USER_PROMPT,
    COVER_LETTER_SYSTEM_PROMPT,
    COVER_LETTER_USER_PROMPT,
    INTERVIEW_PREP_SYSTEM_PROMPT,
    INTERVIEW_PREP_USER_PROMPT,
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
