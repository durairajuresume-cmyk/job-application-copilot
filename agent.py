"""
Lightweight agentic layer over the existing job-matching pipeline.

Implements a think -> act -> observe loop using Claude's native tool-use API.
Given a resume, the agent autonomously decides which tools to call and in
what order — typically: search for jobs, score the top candidates, then draft
a follow-up note for whichever job scores highest.

Tools (each wraps an existing function — no retrieval/scoring logic is
duplicated here):
  1. search_jobs(query)    -> rag.embed_text() + storage.find_matching_jobs()
  2. score_match(job)      -> utils.analyze_resume()
  3. draft_followup(job)   -> new prompt, reuses utils._call_claude()

Guardrail: hard cap of 5 total tool calls per run. Every decision, tool call,
and result is printed to the console as it happens.

Run:
  python agent.py                  # uses the built-in sample resume
  python agent.py path/to/resume.txt
"""

import json
import os
import sys

import anthropic
from dotenv import load_dotenv
from supabase import create_client

from rag import embed_text
from storage import find_matching_jobs
from utils import MODEL, _call_claude, analyze_resume

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

MAX_TOOL_CALLS = 5
MAX_STEPS = 10  # belt-and-suspenders bound on the think/act loop itself

SAMPLE_RESUME = """Priya Sharma
Senior Product Manager, AI Platforms

SUMMARY
Product manager with 6 years of experience shipping AI-powered developer
tools and enterprise AI features. Led a team that built retrieval-augmented
generation (RAG) capabilities and prompt engineering guardrails for an
internal LLM platform used by 2,000+ engineers. Strong background in
agentic workflows, tool-calling architectures, and evaluation frameworks for
model quality and safety.

EXPERIENCE
Senior PM, AI Platform -- Acme Corp (2021-Present)
- Owned roadmap for an internal LLM developer platform, including agent
  orchestration, RAG pipelines, and observability tooling.
- Partnered with ML engineering to define evaluation metrics for
  hallucination rate, latency, and developer adoption.
- Shipped a code-generation assistant used by 500+ engineers, cutting
  review cycle time by 30%.

PM, Conversational AI -- Beta Inc (2018-2021)
- Built a customer-facing chatbot product using retrieval-augmented
  generation and grounding strategies to reduce hallucinations.
- Ran A/B experiments across 1M+ users to improve response quality and
  retention.

SKILLS
LLM product management, RAG, agentic systems, prompt engineering, model
evaluation, developer tools, A/B testing
"""


# ── Tool schemas (given to Claude) ────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_jobs",
        "description": (
            "Semantic search over the shared job postings directory. Returns "
            "up to 10 job postings (title, company, location, similarity) "
            "most similar to the query. Wraps the real embed_text() + "
            "find_matching_jobs() retrieval pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Resume text, or a natural-language description of the target role.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "score_match",
        "description": (
            "Score how well ONE previously-retrieved job matches the "
            "candidate's resume. Returns match_score (0-100), a rationale, "
            "and missing skills. Wraps the real analyze_resume() function. "
            "The job must be one already returned by search_jobs — refer to "
            "it by its exact title."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_title": {
                    "type": "string",
                    "description": "The exact title of a job already returned by search_jobs.",
                }
            },
            "required": ["job_title"],
        },
    },
    {
        "name": "draft_followup",
        "description": (
            "Draft a short outreach/application follow-up note for ONE "
            "previously-retrieved job. The job must be one already returned "
            "by search_jobs — refer to it by its exact title."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_title": {
                    "type": "string",
                    "description": "The exact title of a job already returned by search_jobs.",
                }
            },
            "required": ["job_title"],
        },
    },
]

AGENT_SYSTEM_PROMPT = f"""You are an autonomous job-search agent. Given a candidate's resume, your goal is to:
1. Search the job directory for postings that semantically match the resume.
2. Score the fit of the most promising candidates against the resume.
3. Draft a short outreach/application follow-up note for the single best-scoring job.

You have exactly {MAX_TOOL_CALLS} tool calls total for this run — spend them deliberately. \
A sensible plan is one search_jobs call, two or three score_match calls on the top results, \
and one draft_followup call on whichever job scores highest.

Before each tool call, state in one short sentence what you're about to do and why. \
When you are finished (or if you run out of tool calls), respond with a final plain-text \
summary: the best job found, its match score, and the follow-up note."""


# ── Followup drafting (new logic — not present elsewhere in the codebase) ────

DRAFT_FOLLOWUP_SYSTEM_PROMPT = (
    "You are a job seeker's assistant who writes brief, warm, professional outreach notes."
)

DRAFT_FOLLOWUP_USER_PROMPT = """Write a short outreach/application follow-up note (3-5 sentences, \
under 120 words) for the job below, based on the candidate's resume.

RESUME:
{resume_text}

JOB:
{job_title} at {company}
{description}

Reference 1-2 specific resume qualifications that match this job, and end with a brief, \
confident call to action. Output only the note text, nothing else."""


def draft_followup_note(client: anthropic.Anthropic, resume_text: str, job: dict) -> str:
    prompt = DRAFT_FOLLOWUP_USER_PROMPT.format(
        resume_text=resume_text[:2000],
        job_title=job.get("title", "Untitled"),
        company=job.get("company", "Unknown"),
        description=job.get("description", "")[:800],
    )
    return _call_claude(client, DRAFT_FOLLOWUP_SYSTEM_PROMPT, prompt, max_tokens=300)


# ── Tool execution (the "act" half of think -> act -> observe) ───────────────

class ToolExecutor:
    """Runs each tool against the real pipeline and remembers retrieved jobs
    (by title) so later tool calls can reference a job by name instead of the
    model having to pass the full description back and forth."""

    def __init__(self, supabase, anthropic_client, openai_api_key, resume_text):
        self.supabase = supabase
        self.anthropic_client = anthropic_client
        self.openai_api_key = openai_api_key
        self.resume_text = resume_text
        self.job_memory: dict[str, dict] = {}

    def search_jobs(self, query: str) -> list[dict]:
        query_emb = embed_text(query[:2000], self.openai_api_key)
        matches = find_matching_jobs(self.supabase, query_emb, top_k=10)
        for m in matches:
            self.job_memory[m.get("title", "")] = m
        return [
            {
                "title": m.get("title"),
                "company": m.get("company"),
                "location": m.get("location"),
                "similarity": round(m.get("similarity", 0), 3),
            }
            for m in matches
        ]

    def score_match(self, job_title: str) -> dict:
        job = self.job_memory.get(job_title)
        if job is None:
            return {"error": f"'{job_title}' was not found among previously searched jobs. Call search_jobs first."}
        analysis = analyze_resume(self.anthropic_client, self.resume_text, job.get("description", ""))
        return {
            "job_title": job_title,
            "match_score": analysis.get("match_score"),
            "score_rationale": analysis.get("score_rationale"),
            "missing_skills": analysis.get("missing_skills"),
        }

    def draft_followup(self, job_title: str) -> dict:
        job = self.job_memory.get(job_title)
        if job is None:
            return {"error": f"'{job_title}' was not found among previously searched jobs. Call search_jobs first."}
        note = draft_followup_note(self.anthropic_client, self.resume_text, job)
        return {"job_title": job_title, "followup_note": note}

    def run(self, tool_name: str, tool_input: dict):
        if tool_name == "search_jobs":
            return self.search_jobs(tool_input["query"])
        if tool_name == "score_match":
            return self.score_match(tool_input["job_title"])
        if tool_name == "draft_followup":
            return self.draft_followup(tool_input["job_title"])
        return {"error": f"Unknown tool '{tool_name}'"}


# ── The think -> act -> observe loop ──────────────────────────────────────────

def run_agent(resume_text: str, supabase, anthropic_client) -> None:
    executor = ToolExecutor(supabase, anthropic_client, OPENAI_API_KEY, resume_text)

    print("=" * 78)
    print("AGENTIC JOB MATCHER")
    print(f"Guardrail: capped at {MAX_TOOL_CALLS} tool calls this run.")
    print("=" * 78)

    messages = [
        {
            "role": "user",
            "content": (
                "Here is a candidate's resume:\n\n"
                f"{resume_text}\n\n"
                "Find the best-matching job for them, score the top candidates, "
                "and draft a follow-up note for the best match."
            ),
        }
    ]

    tool_calls_used = 0

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- Step {step} ---")
        tools_available = TOOLS if tool_calls_used < MAX_TOOL_CALLS else []

        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
            tools=tools_available,
            messages=messages,
        )

        text_parts = [block.text for block in response.content if block.type == "text"]
        for text in text_parts:
            if text.strip():
                print(f"[THINK] {text.strip()}")

        if response.stop_reason != "tool_use":
            print("\n" + "=" * 78)
            print("FINAL ANSWER")
            print("=" * 78)
            print(text_parts[0].strip() if text_parts else "(agent returned no text)")
            return

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if tool_calls_used >= MAX_TOOL_CALLS:
                print(f"[GUARDRAIL] Tool call budget ({MAX_TOOL_CALLS}) reached — refusing '{block.name}'.")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": "Tool call budget exhausted. Give your final answer now using only what you've already learned.",
                    "is_error": True,
                })
                continue

            tool_calls_used += 1
            print(f"[ACT {tool_calls_used}/{MAX_TOOL_CALLS}] {block.name}({json.dumps(block.input)})")
            result = executor.run(block.name, block.input)
            print(f"[OBSERVE] {json.dumps(result)[:500]}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

    print("\n[GUARDRAIL] Max step count reached without a final answer.")


def main():
    missing = [
        name for name, val in [
            ("OPENAI_API_KEY", OPENAI_API_KEY),
            ("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY),
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
        ] if not val
    ]
    if missing:
        raise SystemExit(f"Missing required .env values: {', '.join(missing)}")

    resume_path = sys.argv[1] if len(sys.argv) > 1 else None
    if resume_path:
        with open(resume_path) as f:
            resume_text = f.read()
    else:
        resume_text = SAMPLE_RESUME

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    run_agent(resume_text, supabase, anthropic_client)


if __name__ == "__main__":
    main()
