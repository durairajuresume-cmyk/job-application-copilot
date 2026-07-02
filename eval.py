"""
Evaluation harness for the RAG-based Job Matcher.

PART 1 — Retrieval evaluation
  Runs each hand-labeled query through the real pipeline:
    embed_text() [rag.py]  ->  find_matching_jobs() [storage.py]
  and scores the ranked results against a hand-labeled relevant-job set
  with Recall@10, MRR, and NDCG@10.

PART 2 — Generation evaluation
  Runs the real rank_job_matches() [utils.py] explanation generator on the
  retrieved results, then uses a separate Claude call as an LLM judge to
  score groundedness and relevance on a 1-5 scale.

Run:
  python eval.py
"""

import json
import math
import os
import re

import anthropic
from dotenv import load_dotenv
from supabase import create_client

from rag import embed_text
from storage import find_matching_jobs
from utils import rank_job_matches

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

JUDGE_MODEL = "claude-sonnet-4-6"  # same model the app uses for generation
TOP_K = 10


# ── Hand-labeled test set ─────────────────────────────────────────────────────
# "relevant_titles" are exact job titles from seed_jobs.py's JOBS list — the
# stable identifier we agreed on, since job_postings.id is a DB-generated UUID
# that doesn't exist until a job is seeded.

TEST_SET = [
    {
        "id": "Q1_healthcare_ai_pm",
        "resume_text": (
            "Product manager with 5 years of experience leading AI features for "
            "electronic health record platforms, including clinical note "
            "summarization using LLMs and predictive readmission risk models. "
            "Deep experience navigating HIPAA and FDA regulatory requirements for "
            "AI in clinical settings, partnering with physicians and clinical "
            "informaticists to ensure AI outputs are safe and explainable."
        ),
        "relevant_titles": {
            "AI Product Manager – Healthcare",
            "Product Manager – AI Health Diagnostics",
        },
    },
    {
        "id": "Q2_conversational_ai_pm",
        "resume_text": (
            "PM with 4+ years building conversational AI and enterprise chatbot "
            "experiences, including retrieval augmented generation (RAG) and "
            "grounding strategies to reduce hallucinations. Shipped AI writing "
            "assistants and customer-facing conversational UI features, running "
            "A/B experiments to improve retention and response quality across "
            "millions of users."
        ),
        "relevant_titles": {
            "Product Manager – Conversational AI",
            "Product Manager – Conversational UI",
            "Senior PM – AI Writing Assistant",
        },
    },
    {
        "id": "Q3_ml_platform_pm",
        "resume_text": (
            "Senior PM for internal machine learning platforms, owning roadmaps "
            "for feature stores, model training pipelines, and model serving "
            "infrastructure used by hundreds of data scientists. Deep understanding "
            "of MLOps, distributed systems, and the full ML lifecycle from data "
            "preparation to production deployment."
        ),
        "relevant_titles": {
            "Senior PM – ML Platform",
            "Product Manager – ML Platform",
            "Product Manager – Data Platform",
            "Senior PM – AI Research Platform",
        },
    },
    {
        "id": "Q4_computer_vision_pm",
        "resume_text": (
            "Product manager focused on computer vision and autonomous systems, "
            "working closely with perception and sensor fusion engineering teams "
            "to ship self-driving and camera-based safety features. Experience "
            "defining evaluation metrics for model accuracy in real-world driving "
            "and manufacturing environments."
        ),
        "relevant_titles": {
            "Product Manager – Autonomous Systems",
            "Senior PM – Computer Vision Platform",
            "Senior PM – Computer Vision AI",
        },
    },
    {
        "id": "Q5_ai_safety_pm",
        "resume_text": (
            "PM at the intersection of AI safety research and product, translating "
            "alignment and interpretability findings into product constraints and "
            "UI guardrails. Built evaluation frameworks for model harmlessness, "
            "bias, and fairness, and managed roadmaps for responsible AI and trust "
            "& safety features."
        ),
        "relevant_titles": {
            "Product Manager – AI Safety",
            "Senior PM – Responsible AI",
            "Product Manager – AI Trust & Safety",
        },
    },
    {
        "id": "Q6_genai_creative_pm",
        "resume_text": (
            "Product lead for generative AI creative tools, shipping text-to-image, "
            "generative fill, and video generation features for creative "
            "professionals. Deep knowledge of diffusion models and strong design "
            "sensibility, partnering with design teams to ship features used by "
            "millions of creators."
        ),
        "relevant_titles": {
            "Director of Product – Generative AI",
            "Product Manager – AI Design Tools",
            "Product Manager – Generative Media AI",
            "Senior PM – AI-First Design Features",
        },
    },
    {
        "id": "Q7_ai_agents_devtools_pm",
        "resume_text": (
            "PM for developer-facing AI products, shipping agentic workflows, AI "
            "code generation, and LLM developer platforms used by thousands of "
            "engineers. Experience with prompt engineering, tool-calling "
            "architectures, and building infrastructure that lets developers "
            "compose autonomous AI agents."
        ),
        "relevant_titles": {
            "Product Manager – AI Agents",
            "Senior PM – Agentic AI Products",
            "Product Manager – AI Copilot Features",
            "Product Manager – AI Code Generation",
        },
    },
    {
        "id": "Q8_enterprise_search_pm",
        "resume_text": (
            "Senior PM for enterprise search and knowledge management products, "
            "building semantic search over internal company knowledge bases and "
            "improving search/discovery relevance at scale. Partnered with ML "
            "engineering to tune ranking and retrieval quality for large "
            "enterprise customers."
        ),
        "relevant_titles": {
            "Senior PM – Enterprise Search",
            "Senior Product Manager – Search & Discovery",
            "Senior PM – Knowledge Management AI",
            "Senior PM – AI Features",
        },
    },
    {
        "id": "Q9_regulated_enterprise_ai_pm",
        "resume_text": (
            "PM shipping AI features for regulated enterprise domains including "
            "finance, legal, and HR, ensuring compliance with industry regulations "
            "while automating document review and financial research workflows. "
            "Experience partnering with legal and compliance teams to ship AI "
            "responsibly in high-stakes environments."
        ),
        "relevant_titles": {
            "Product Manager – AI-Powered Finance",
            "Senior PM – AI Legal Tools",
            "Senior PM – AI Financial Research",
            "Senior PM – AI-Powered HR",
        },
    },
    {
        "id": "Q10_voice_multimodal_pm",
        "resume_text": (
            "PM for voice interfaces and multimodal AI products, shipping speech "
            "recognition and multimodal LLM features that combine text, image, and "
            "audio understanding. Experience defining evaluation frameworks for "
            "speech accuracy and multimodal reasoning quality."
        ),
        "relevant_titles": {
            "Senior PM – Voice AI",
            "Senior PM – Multimodal AI",
            "AI Product Lead",
        },
    },
]


# ── Part 1: retrieval metrics ─────────────────────────────────────────────────

def recall_at_k(retrieved_titles: list[str], relevant_titles: set[str], k: int = TOP_K) -> float:
    if not relevant_titles:
        return 0.0
    hits = len(set(retrieved_titles[:k]) & relevant_titles)
    return hits / len(relevant_titles)


def reciprocal_rank(retrieved_titles: list[str], relevant_titles: set[str], k: int = TOP_K) -> float:
    for i, title in enumerate(retrieved_titles[:k], start=1):
        if title in relevant_titles:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_titles: list[str], relevant_titles: set[str], k: int = TOP_K) -> float:
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, title in enumerate(retrieved_titles[:k], start=1)
        if title in relevant_titles
    )
    ideal_hits = min(len(relevant_titles), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def run_retrieval_eval(supabase, openai_api_key: str) -> list[dict]:
    print("\n" + "=" * 78)
    print("PART 1 — RETRIEVAL EVALUATION (embed_text -> find_matching_jobs)")
    print("=" * 78)

    results = []
    for case in TEST_SET:
        query_emb = embed_text(case["resume_text"][:2000], openai_api_key)
        matches = find_matching_jobs(supabase, query_emb, top_k=TOP_K)
        retrieved_titles = [m.get("title", "") for m in matches]

        recall = recall_at_k(retrieved_titles, case["relevant_titles"])
        rr = reciprocal_rank(retrieved_titles, case["relevant_titles"])
        ndcg = ndcg_at_k(retrieved_titles, case["relevant_titles"])

        print(f"\n[{case['id']}]")
        print(f"  Relevant set: {sorted(case['relevant_titles'])}")
        print("  Retrieved (rank: title @ company — similarity):")
        for i, m in enumerate(matches, start=1):
            hit = "✓" if m.get("title", "") in case["relevant_titles"] else " "
            print(
                f"    {hit} {i:>2}. {m.get('title', 'Untitled')} @ "
                f"{m.get('company', 'Unknown')} — sim={m.get('similarity', 0):.3f}"
            )
        print(f"  Recall@{TOP_K}={recall:.3f}  RR={rr:.3f}  NDCG@{TOP_K}={ndcg:.3f}")

        results.append({
            "id": case["id"],
            "matches": matches,
            "retrieved_titles": retrieved_titles,
            "recall": recall,
            "rr": rr,
            "ndcg": ndcg,
        })

    print("\n" + "-" * 78)
    print("RETRIEVAL SUMMARY")
    print("-" * 78)
    header = f"{'query':<28}{'Recall@10':>12}{'RR':>10}{'NDCG@10':>10}"
    print(header)
    for r in results:
        print(f"{r['id']:<28}{r['recall']:>12.3f}{r['rr']:>10.3f}{r['ndcg']:>10.3f}")

    n = len(results)
    mean_recall = sum(r["recall"] for r in results) / n
    mean_mrr = sum(r["rr"] for r in results) / n
    mean_ndcg = sum(r["ndcg"] for r in results) / n
    print("-" * 78)
    print(f"{'MEAN':<28}{mean_recall:>12.3f}{mean_mrr:>10.3f}{mean_ndcg:>10.3f}")

    return results


# ── Part 2: generation evaluation (LLM-as-judge) ──────────────────────────────

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of AI-generated job-match explanations. You "
    "check claims against source data only — you do not use outside knowledge "
    "about companies or roles. You always return valid JSON."
)

JUDGE_USER_PROMPT = """A job-matching system retrieved the job postings below for a candidate, then \
generated a ranked explanation of fit. Evaluate the generated explanation.

CANDIDATE RESUME:
{resume_text}

RETRIEVED JOB POSTINGS (the ONLY source of truth the explanation may draw on):
{postings_block}

GENERATED EXPLANATION TO EVALUATE:
{generated_text}

Score the generated explanation on two dimensions, 1-5 each:

a) groundedness: Is every claim about a job (title, company, requirements, \
responsibilities) actually supported by the retrieved postings above, with no \
hallucinated or invented details? 5 = fully grounded, 1 = frequent fabrication.

b) relevance: Does the explanation correctly and specifically justify why each \
job does or doesn't fit THIS resume (not generic boilerplate)? 5 = precise, \
resume-specific reasoning throughout, 1 = generic or incorrect justification.

Return ONLY valid JSON in exactly this structure (no markdown, no extra text):
{{
  "groundedness": <integer 1-5>,
  "groundedness_reasoning": "<1-2 sentences, cite a specific claim if ungrounded>",
  "relevance": <integer 1-5>,
  "relevance_reasoning": "<1-2 sentences>"
}}"""


def judge_generation(client: anthropic.Anthropic, resume_text: str, matches: list, generated_text: str) -> dict:
    postings_block = "\n\n".join(
        f"[{i + 1}] {m.get('title', 'Untitled')} at {m.get('company', 'Unknown')}\n"
        f"{m.get('description', '')[:500]}"
        for i, m in enumerate(matches)
    )
    prompt = JUDGE_USER_PROMPT.format(
        resume_text=resume_text,
        postings_block=postings_block,
        generated_text=generated_text,
    )
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=500,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = message.content[0].text
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not json_match:
        raise ValueError(f"No JSON object found in judge response:\n{raw}")
    return json.loads(json_match.group())


def run_generation_eval(client: anthropic.Anthropic, retrieval_results: list[dict]) -> list[dict]:
    print("\n" + "=" * 78)
    print("PART 2 — GENERATION EVALUATION (rank_job_matches + LLM judge)")
    print("=" * 78)

    case_by_id = {c["id"]: c for c in TEST_SET}
    results = []
    for r in retrieval_results:
        case = case_by_id[r["id"]]
        matches = r["matches"]

        if not matches:
            print(f"\n[{r['id']}] skipped — no retrieved jobs to generate an explanation for")
            continue

        generated_text = rank_job_matches(client, case["resume_text"], matches)
        judgment = judge_generation(client, case["resume_text"], matches, generated_text)

        print(f"\n[{r['id']}]")
        print("  Generated explanation (rank_job_matches):")
        for line in generated_text.splitlines():
            print(f"    {line}")
        print(
            f"  Judge — groundedness: {judgment['groundedness']}/5 "
            f"({judgment['groundedness_reasoning']})"
        )
        print(
            f"  Judge — relevance:    {judgment['relevance']}/5 "
            f"({judgment['relevance_reasoning']})"
        )

        results.append({
            "id": r["id"],
            "groundedness": judgment["groundedness"],
            "relevance": judgment["relevance"],
        })

    print("\n" + "-" * 78)
    print("GENERATION SUMMARY")
    print("-" * 78)
    header = f"{'query':<28}{'Groundedness':>14}{'Relevance':>12}"
    print(header)
    for r in results:
        print(f"{r['id']:<28}{r['groundedness']:>14}{r['relevance']:>12}")

    n = len(results)
    if n:
        mean_grounded = sum(r["groundedness"] for r in results) / n
        mean_relevance = sum(r["relevance"] for r in results) / n
        print("-" * 78)
        print(f"{'MEAN':<28}{mean_grounded:>14.2f}{mean_relevance:>12.2f}")

    return results


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

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    retrieval_results = run_retrieval_eval(supabase, OPENAI_API_KEY)
    run_generation_eval(anthropic_client, retrieval_results)


if __name__ == "__main__":
    main()
