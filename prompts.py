ANALYSIS_SYSTEM_PROMPT = """You are an expert career coach and ATS (Applicant Tracking System) specialist \
with deep knowledge of hiring practices across industries. Your analysis is precise, honest, and actionable."""

ANALYSIS_USER_PROMPT = """Analyze the following resume against the job description and return a structured assessment.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Return ONLY valid JSON in exactly this structure (no markdown, no extra text):
{{
  "match_score": <integer 0-100>,
  "score_rationale": "<one sentence explaining the score>",
  "strengths": [
    "<specific strength that aligns with the job>",
    "<specific strength that aligns with the job>"
  ],
  "missing_skills": [
    "<skill, qualification, or experience the candidate lacks>",
    "<skill, qualification, or experience the candidate lacks>"
  ],
  "improvements": [
    "<concrete, actionable resume improvement for this role>",
    "<concrete, actionable resume improvement for this role>"
  ]
}}

Rules:
- match_score: weight skills match 40%, experience level 30%, keywords/ATS 20%, education 10%
- strengths: 4–6 items that directly map to job requirements
- missing_skills: 3–6 gaps that matter most for this role
- improvements: 4–6 specific rewrites or additions (e.g. "Add a metrics-driven bullet for X project")"""


COVER_LETTER_SYSTEM_PROMPT = """You are an expert career coach who writes compelling, \
personalized cover letters. Your letters are professional, concise, and tailored to the specific role."""

COVER_LETTER_USER_PROMPT = """Write a tailored cover letter for the position described below, \
using the candidate's resume as the source of truth for experience and skills.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Requirements:
- Start with "Dear Hiring Manager,"
- 3–4 focused paragraphs, 250–350 words total
- Paragraph 1: Hook — name the role and lead with the candidate's strongest relevant qualification
- Paragraph 2–3: Connect 2–3 specific resume experiences directly to key job requirements
- Final paragraph: Confident call to action; use [Company Name] if the company is not specified
- Tone: Professional, direct, and personable — not generic or sycophantic

Output only the cover letter text, nothing else."""


INTERVIEW_PREP_SYSTEM_PROMPT = """You are an expert interview coach who prepares candidates \
with targeted, role-specific interview strategies."""

INTERVIEW_PREP_USER_PROMPT = """Create comprehensive interview preparation materials for the candidate below.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

Structure your response with these exact sections:

## Likely Technical / Skills Questions
1. [question]
2. [question]
3. [question]

## Likely Behavioral Questions
1. [question]
2. [question]
3. [question]

## Suggested STAR Answers
For each behavioral question above, provide a brief STAR outline drawn from the candidate's actual experience.

## Key Talking Points
- [strength or theme to weave throughout the interview]
- [strength or theme to weave throughout the interview]
- [strength or theme to weave throughout the interview]

## Questions to Ask the Interviewer
1. [thoughtful question that shows research]
2. [thoughtful question about the role]
3. [thoughtful question about growth/team]

## Potential Red Flags and How to Address Them
- [gap or concern] → [how to frame it positively]
- [gap or concern] → [how to frame it positively]"""
