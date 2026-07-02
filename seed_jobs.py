"""
Seed 50 sample AI/tech PM job postings into Supabase.

WHY THIS EXISTS:
  The Job Matcher's semantic search needs a corpus of postings to search against.
  This script embeds 50 realistic job descriptions and inserts them so you can
  immediately test vector similarity without manually adding postings one by one.

PREREQUISITES:
  1. Run migrations/001_pgvector_job_postings.sql in Supabase SQL Editor first.
  2. Add to your .env file:
       SUPABASE_SERVICE_KEY=<your service_role key>
         → Supabase Dashboard → Settings → API → "service_role" (Secret) key
         → This key bypasses Row Level Security so the script can insert as admin
       SEED_USER_ID=<your user UUID>
         → Supabase Dashboard → Authentication → Users → click your email → copy UUID
  3. Your SUPABASE_URL and OPENAI_API_KEY should already be in .env.

RUN:
  python seed_jobs.py

COST:
  50 embeddings × ~200 tokens each ≈ 10,000 tokens ≈ $0.002 (fractions of a cent).
"""

import os
import sys
import time

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SEED_USER_ID = os.getenv("SEED_USER_ID", "")

# ── 50 sample job postings ────────────────────────────────────────────────────
# Covering AI PM, technical PM, platform PM, and data PM roles across industries.
# Descriptions are intentionally varied in vocabulary to test semantic search:
# some say "machine learning", others "AI/ML", others "intelligent systems" — the
# embeddings should surface them all when your resume mentions any of these.

JOBS = [
    {
        "title": "Senior AI Product Manager",
        "company": "Google DeepMind",
        "location": "London, UK",
        "description": (
            "Lead product strategy for our frontier AI research platform used by thousands of "
            "researchers globally. You'll define the roadmap for tools that accelerate scientific "
            "discovery using large language models and multimodal AI. Responsibilities include "
            "translating cutting-edge research breakthroughs into user-facing product features, "
            "working closely with ML researchers, and defining success metrics for experimental "
            "AI capabilities. The ideal candidate has 5+ years of product management experience, "
            "deep familiarity with the AI/ML research ecosystem, and a track record of shipping "
            "developer tools or research infrastructure at scale. Experience with LLMs, "
            "diffusion models, or reinforcement learning is a strong plus. You'll own the "
            "product vision for a suite of internal tools that make AI research faster and more "
            "rigorous, including experiment tracking, dataset management, and model evaluation "
            "frameworks. This is a high-impact role at the frontier of AI development."
        ),
    },
    {
        "title": "Product Manager – Conversational AI",
        "company": "Microsoft",
        "location": "Redmond, WA",
        "description": (
            "Drive the product roadmap for Copilot's conversational AI features embedded across "
            "Office 365, Teams, and Outlook. You will work with large language model teams to "
            "design grounding, retrieval augmented generation (RAG), and context management "
            "strategies that make Copilot more helpful and accurate. Core responsibilities: "
            "defining user stories for enterprise AI assistants, setting quality and safety "
            "guardrails with policy teams, and running A/B experiments to measure retention. "
            "Requires 4+ years in product management, experience with NLP or conversational "
            "products, and comfort analyzing telemetry data from millions of daily users. "
            "Familiarity with prompt engineering, hallucination mitigation, and responsible AI "
            "principles is expected. You'll partner closely with engineering, design, and "
            "legal/compliance teams to ship AI features to 300M+ enterprise users worldwide."
        ),
    },
    {
        "title": "Senior PM – ML Platform",
        "company": "Uber",
        "location": "San Francisco, CA",
        "description": (
            "Own the roadmap for Michelangelo, Uber's internal machine learning platform "
            "serving thousands of ML engineers. Your customers are internal data scientists "
            "building models for surge pricing, fraud detection, ETAs, and driver matching. "
            "You will prioritize infrastructure investments across feature stores, model "
            "training pipelines, model serving, and monitoring. Strong technical background "
            "required — you should understand distributed systems, MLOps practices, and the "
            "full ML lifecycle from data preparation to production deployment. The role "
            "demands 5+ years of PM experience, ideally in developer tools or ML infrastructure. "
            "You'll define success metrics like model deployment frequency, p99 inference "
            "latency, and engineer productivity. Partners include ML engineering, data "
            "engineering, and applied science teams across 50+ business units."
        ),
    },
    {
        "title": "Director of Product – Generative AI",
        "company": "Adobe",
        "location": "San Jose, CA",
        "description": (
            "Lead a team of PMs building Adobe Firefly, our generative AI suite for creative "
            "professionals. This director-level role sets product strategy for text-to-image, "
            "generative fill, video generation, and vector generation capabilities. You'll "
            "manage 3-5 PMs and represent product in cross-functional leadership forums. "
            "Deep knowledge of the generative AI landscape (diffusion models, GANs, "
            "fine-tuning) and a strong design sensibility are essential. Prior experience "
            "shipping creative software or working with creative professionals is a major plus. "
            "Success metrics include DAU/MAU for Firefly features, content quality scores "
            "from user research, and revenue impact via Creative Cloud subscriptions. "
            "7+ years of product management experience required; 2+ years managing PMs."
        ),
    },
    {
        "title": "AI Product Manager – Healthcare",
        "company": "Epic Systems",
        "location": "Verona, WI",
        "description": (
            "Define AI-powered features for Epic's electronic health record platform used by "
            "hospitals serving 350M patients. Current AI initiatives include clinical note "
            "summarization using LLMs, predictive readmission risk models, ambient clinical "
            "documentation, and diagnostic imaging AI. You will work with clinical informaticists, "
            "physicians, and ML engineers to ensure AI outputs are accurate, safe, and "
            "explainable. Healthcare AI requires rigorous attention to bias, fairness, and FDA "
            "regulatory considerations — familiarity with these is required. 4+ years PM "
            "experience; clinical informatics, health IT, or digital health background strongly "
            "preferred. You'll define evaluation frameworks for AI safety, manage stakeholder "
            "relationships with hospital CIOs, and drive a roadmap aligned with HIPAA and "
            "ONC compliance requirements."
        ),
    },
    {
        "title": "Product Manager – AI Safety",
        "company": "Anthropic",
        "location": "San Francisco, CA",
        "description": (
            "Shape how Claude's safety capabilities are surfaced to users and developers. "
            "This role sits at the intersection of AI safety research and product — you'll "
            "work with alignment researchers to translate safety findings into product "
            "constraints, UI guardrails, and API affordances. Responsibilities include defining "
            "the developer experience for Constitutional AI, building evaluation dashboards for "
            "model harmlessness, and managing the roadmap for safety-related API features like "
            "content moderation and usage policies. Deep intellectual curiosity about AI risks "
            "and a strong ethical compass are non-negotiable. 4+ years in product; background "
            "in policy, philosophy, or ML research is a plus. You'll regularly engage with "
            "external researchers, government stakeholders, and enterprise customers on "
            "responsible AI deployment."
        ),
    },
    {
        "title": "Senior Product Manager – Recommendation Systems",
        "company": "Netflix",
        "location": "Los Gatos, CA",
        "description": (
            "Own the product roadmap for Netflix's personalization and recommendation engine "
            "that determines what 260M subscribers see on their homepage. You'll work with "
            "research scientists building two-tower models, transformer-based rankers, and "
            "contextual bandit systems for exploration. Responsibilities include defining "
            "business objectives translated into ML optimization targets, running large-scale "
            "A/B tests, and measuring causal impact of recommendation changes on viewing hours "
            "and subscriber retention. Strong data intuition required — you'll regularly write "
            "SQL and interpret statistical significance. 5+ years of PM experience; prior "
            "work in recommendation systems, ads, or search ranking is preferred. Partners "
            "include ML scientists, data engineers, and the content strategy team."
        ),
    },
    {
        "title": "Product Manager – NLP Platform",
        "company": "Hugging Face",
        "location": "Remote",
        "description": (
            "Define the product experience for Hugging Face Hub, the world's largest "
            "repository of open-source ML models, datasets, and Spaces. Your customers are "
            "ML engineers and researchers who use the Hub to discover, fine-tune, and deploy "
            "models. Core focus areas: model card discoverability, dataset viewer UX, "
            "inference API experience, and enterprise collaboration features. Strong developer "
            "empathy is essential — you should understand the friction points of fine-tuning "
            "transformers, running inference, and managing model versions. 3+ years PM "
            "experience, ideally in open-source developer tools or NLP applications. "
            "Familiarity with PyTorch, the transformers library, and the OSS ML ecosystem "
            "is a major plus. Remote-first culture; quarterly team gatherings in NYC and Paris."
        ),
    },
    {
        "title": "Senior PM – Fraud Detection AI",
        "company": "Stripe",
        "location": "San Francisco, CA",
        "description": (
            "Lead the product roadmap for Stripe Radar, our machine learning fraud prevention "
            "product protecting billions of dollars in payment volume. Radar uses gradient "
            "boosted trees, graph neural networks, and real-time feature computation to score "
            "transactions in under 100ms. You'll define new ML signals, manage the tradeoff "
            "between false positives (declined legitimate purchases) and false negatives "
            "(approved fraud), and build features that give merchants control over risk "
            "tolerance. 5+ years PM experience; risk, payments, or fintech background "
            "strongly preferred. You'll partner with data scientists, payments engineers, and "
            "merchant success teams. Success metrics include fraud rate, chargeback rate, "
            "and legitimate decline rate across different merchant verticals."
        ),
    },
    {
        "title": "Product Manager – AI Copilot Features",
        "company": "GitHub",
        "location": "Remote",
        "description": (
            "Own GitHub Copilot's next-generation code completion and chat features used by "
            "1M+ developers daily. You'll define how Copilot uses retrieval augmented "
            "generation (RAG) over repository context, manages prompt windows, and adapts "
            "suggestions to individual coding style. Responsibilities include running "
            "developer research, defining acceptance/code quality metrics, and managing the "
            "roadmap for Copilot Chat, Copilot in PRs, and Copilot for CLI. Deep understanding "
            "of software development workflows is required — you should be comfortable reading "
            "code and understanding developer pain points. 4+ years PM experience; developer "
            "tools or IDE products background preferred. Partners include LLM fine-tuning "
            "teams, VS Code engineering, and the Enterprise product team."
        ),
    },
    {
        "title": "Senior PM – AI Writing Assistant",
        "company": "Grammarly",
        "location": "San Francisco, CA",
        "description": (
            "Drive the product vision for Grammarly's generative AI writing features including "
            "GrammarlyGO and context-aware tone suggestions. This role owns the end-to-end "
            "experience of helping 30M daily users write better with AI assistance — from "
            "detecting writing intent to generating on-brand suggestions across email, docs, "
            "and social media. You'll work with LLM researchers on few-shot prompting, "
            "personalization models, and style transfer. Strong user research skills and a "
            "passion for language are essential. 4+ years PM experience; prior B2C product "
            "management or consumer AI product experience preferred. You'll own the roadmap "
            "for Grammarly's largest new surface area and partner with growth, monetization, "
            "and enterprise teams."
        ),
    },
    {
        "title": "Product Manager – Autonomous Systems",
        "company": "Waymo",
        "location": "Mountain View, CA",
        "description": (
            "Define product requirements for Waymo's autonomous driving software stack, "
            "focusing on perception, prediction, and planning systems that allow the vehicle "
            "to navigate complex urban environments. You'll translate safety-critical "
            "performance requirements into engineering specifications, manage the roadmap for "
            "specific operational design domains (ODDs), and work with simulation teams to "
            "define scenario coverage for testing. Comfort with systems thinking and an "
            "ability to reason about probabilistic systems under uncertainty are essential. "
            "5+ years PM experience; robotics, autonomous systems, or safety-critical "
            "software background required. Partners include ML, software, hardware, and "
            "operations teams. Deep familiarity with AV evaluation frameworks (disengagements, "
            "collision rate, comfortable miles) expected."
        ),
    },
    {
        "title": "AI Product Manager – Customer Intelligence",
        "company": "Salesforce",
        "location": "San Francisco, CA",
        "description": (
            "Build AI-powered insights features for Salesforce Einstein, helping sales reps "
            "close deals faster with intelligent pipeline analysis, lead scoring, and "
            "next-best-action recommendations. This role owns the roadmap for predictive "
            "analytics features in Sales Cloud, including opportunity health scores, churn "
            "prediction, and AI-generated call summaries via Einstein Conversation Insights. "
            "4+ years PM experience; CRM, sales tech, or B2B SaaS background preferred. "
            "Strong data analysis skills required — you'll regularly analyze feature usage "
            "metrics, define A/B test designs, and work with data scientists building "
            "classification and regression models. You'll partner with Sales Cloud "
            "engineering, Einstein Platform, and the AppExchange marketplace team."
        ),
    },
    {
        "title": "Senior Product Manager – Search & Discovery",
        "company": "Airbnb",
        "location": "San Francisco, CA",
        "description": (
            "Own the search and discovery experience for Airbnb's global marketplace of 7M+ "
            "listings. This includes the ranking algorithm that determines listing order, "
            "semantic search that understands traveler intent, and discovery surfaces like "
            "Categories and Explore. You'll work with ML engineers building dense retrieval "
            "models, learning-to-rank systems, and natural language understanding for search "
            "queries. Success metrics include booking conversion rate, search abandonment, "
            "and listing page views. 5+ years PM experience; search, marketplace, or "
            "recommendation systems background required. Strong SQL skills and comfort with "
            "causal inference (not just correlation) are expected. Partners include ranking "
            "ML, trust & safety, and the host experience team."
        ),
    },
    {
        "title": "Product Lead – AI Research Tools",
        "company": "Perplexity AI",
        "location": "San Francisco, CA",
        "description": (
            "Shape the future of AI-native search at Perplexity, where every answer is "
            "grounded in real-time web sources and cited for accuracy. You'll own the product "
            "roadmap for Perplexity Pro, our research-grade assistant used by academics, "
            "journalists, and professionals. Core areas: improving answer quality through "
            "better retrieval, reducing hallucinations via source grounding, and building "
            "agentic research flows that can plan multi-step investigations. 4+ years PM "
            "experience; information retrieval, media, or consumer AI product background "
            "preferred. You should deeply understand RAG architecture, citation accuracy "
            "metrics, and the nuances of web crawling and freshness. This is a high-autonomy "
            "role in a fast-moving team of 50 people."
        ),
    },
    {
        "title": "Senior PM – Voice AI",
        "company": "Apple",
        "location": "Cupertino, CA",
        "description": (
            "Drive product strategy for Siri's next generation of on-device and server-side "
            "language understanding capabilities. You'll work at the intersection of speech "
            "recognition, natural language processing, and on-device ML — a unique combination "
            "requiring technical depth and user empathy. Responsibilities include defining "
            "intent recognition requirements, managing the roadmap for Siri Shortcuts and "
            "App Intents integration, and driving quality metrics for multilingual support. "
            "6+ years PM experience required; voice interfaces, NLP, or on-device ML "
            "background essential. Privacy-first product thinking is non-negotiable. "
            "You'll partner with Siri Language, Core ML, and the developer ecosystem team. "
            "Apple-style operational rigor and confidentiality are expected."
        ),
    },
    {
        "title": "Product Manager – Personalization Engine",
        "company": "Spotify",
        "location": "New York, NY",
        "description": (
            "Own the roadmap for Spotify's personalization features — Discover Weekly, "
            "Daily Mixes, and the AI DJ — which drive listening hours for 600M users. "
            "You'll work with collaborative filtering, content-based embedding models, and "
            "large language models that generate playlist narratives and transitions. "
            "Responsibilities include defining listening quality metrics, running global "
            "A/B tests, and managing the product roadmap in collaboration with ML research "
            "and editorial teams. Strong analytical background required; experience with "
            "recommendation systems, media, or consumer subscription products preferred. "
            "4+ years PM experience. You'll advocate for user listening satisfaction while "
            "balancing creator and label relationships."
        ),
    },
    {
        "title": "AI Product Lead",
        "company": "Cohere",
        "location": "Toronto, ON",
        "description": (
            "Define the product experience for Cohere's enterprise NLP platform — Command, "
            "Embed, and Rerank — used by Fortune 500 companies to build AI-powered search, "
            "classification, and generation applications. You'll work directly with enterprise "
            "customers to understand use cases, translate needs into platform features, and "
            "drive the go-to-market roadmap for new model capabilities. Deep understanding "
            "of LLM evaluation, fine-tuning workflows, and enterprise deployment patterns "
            "(on-prem, cloud, VPC) is expected. 4+ years PM experience; B2B AI/ML platform "
            "or API product background required. Partners include sales engineering, research, "
            "and the developer relations team. You'll represent the customer voice in research "
            "prioritization discussions."
        ),
    },
    {
        "title": "Senior PM – Enterprise Search",
        "company": "Glean",
        "location": "Palo Alto, CA",
        "description": (
            "Build the world's best work search engine at Glean, which uses LLMs and "
            "semantic retrieval to help employees find information across Slack, Google Drive, "
            "Jira, Confluence, and 100+ other enterprise tools. You'll own the product "
            "roadmap for core search quality, semantic ranking, and the AI assistant that "
            "answers questions from enterprise knowledge bases. Deep understanding of "
            "information retrieval, embeddings, and enterprise SaaS integrations is "
            "required. 4+ years PM experience; enterprise software, B2B SaaS, or "
            "information retrieval background preferred. You'll work directly with "
            "CIOs and IT leaders at Fortune 1000 companies to understand knowledge "
            "management challenges and drive adoption."
        ),
    },
    {
        "title": "Product Manager – Data Platform",
        "company": "Databricks",
        "location": "San Francisco, CA",
        "description": (
            "Own the product roadmap for Databricks Unity Catalog and data governance "
            "features used by data engineers and ML practitioners at 10,000+ enterprises. "
            "You'll define the experience for data discovery, lineage tracking, access "
            "controls, and AI-powered data quality monitoring. Strong data engineering "
            "background required — you should understand Delta Lake, Apache Spark, and "
            "the modern data stack. 4+ years PM experience in data platforms, analytics "
            "engineering, or MLOps tooling. Partners include the Lakehouse platform team, "
            "ML runtime team, and enterprise field engineering. This role has significant "
            "influence on how AI/ML teams govern and share data assets across large "
            "organizations."
        ),
    },
    {
        "title": "Senior PM – AI Features",
        "company": "Notion",
        "location": "San Francisco, CA",
        "description": (
            "Drive the roadmap for Notion AI — the embedded AI assistant used by 30M+ "
            "users to write, summarize, translate, and organize information. You'll manage "
            "features including AI writing, smart summarization of long documents, AI search "
            "across workspaces, and the AI blocks API for developers. Deeply user-centric "
            "role — you'll conduct regular user research to understand how knowledge workers "
            "want AI to assist their thinking without replacing it. 4+ years PM experience; "
            "productivity software, writing tools, or B2B SaaS background preferred. "
            "Partners include the AI engineering team, growth, and developer relations. "
            "You'll define quality metrics for AI output (accuracy, tone, usefulness) and "
            "oversee the responsible AI guidelines for Notion AI."
        ),
    },
    {
        "title": "Senior PM – Conversational Commerce",
        "company": "Shopify",
        "location": "Remote",
        "description": (
            "Define AI-powered shopping experiences including Shopify's AI product "
            "recommendations, merchant AI assistant, and natural language storefront search. "
            "You'll work with LLM and recommendation system teams to help merchants grow "
            "their businesses with intelligent insights: inventory forecasting, customer "
            "segmentation, and automated marketing copy generation. 5+ years PM experience; "
            "e-commerce, marketplace, or B2B SaaS background strongly preferred. "
            "Understanding of merchant economics, conversion optimization, and the Shopify "
            "ecosystem (apps, payments, checkout) is a plus. Remote-first role; quarterly "
            "in-person sprints. Partners include the merchant experience team, Shopify AI "
            "research, and platform engineering."
        ),
    },
    {
        "title": "Product Manager – AI Analytics",
        "company": "Tableau (Salesforce)",
        "location": "Seattle, WA",
        "description": (
            "Own the AI features roadmap for Tableau, including Tableau Pulse (AI-generated "
            "data narratives), Einstein Analytics natural language querying, and automated "
            "anomaly detection. You'll help business users ask questions of their data in "
            "plain English and get visual, explainable answers — no SQL required. "
            "4+ years PM experience; analytics, business intelligence, or data visualization "
            "background required. Strong understanding of how business analysts use data and "
            "what makes AI-generated insights trustworthy. Partners include the data science "
            "team, Salesforce Einstein AI, and enterprise field teams. Success metrics "
            "include engagement with AI features, time-to-insight, and user confidence in "
            "AI-generated narratives."
        ),
    },
    {
        "title": "AI Product Manager",
        "company": "Scale AI",
        "location": "San Francisco, CA",
        "description": (
            "Build the product roadmap for Scale's data labeling and evaluation platform "
            "used by leading AI labs and enterprises to train and evaluate frontier models. "
            "You'll work on products that enable efficient human-in-the-loop data annotation, "
            "RLHF (reinforcement learning from human feedback) data pipelines, and model "
            "evaluation benchmarking. Deep understanding of the ML training data pipeline "
            "is essential. 4+ years PM experience; AI/ML tooling, data operations, or "
            "enterprise B2B background preferred. You'll engage directly with technical "
            "customers (ML researchers at AI labs) and ensure Scale's platform meets the "
            "quality and scale requirements of frontier model training. Partners include "
            "labeling operations, ML infrastructure, and research teams."
        ),
    },
    {
        "title": "Senior PM – AI Research Platform",
        "company": "Weights & Biases",
        "location": "San Francisco, CA",
        "description": (
            "Own the product roadmap for W&B's MLOps platform — experiment tracking, "
            "model registry, hyperparameter sweeps, and dataset versioning — used by "
            "500,000+ ML engineers and researchers globally. You'll work closely with "
            "the ML community to understand how practitioners want to iterate on model "
            "development and collaborate with their teams. Recent initiatives include "
            "LLM evaluation tools, prompt management, and integration with popular "
            "fine-tuning frameworks. 4+ years PM experience; developer tools, MLOps, "
            "or ML research background strongly preferred. You'll act as a translator "
            "between ML practitioner needs and engineering priorities. Strong community "
            "instincts and comfort in technical forums (Slack, Discord, GitHub) expected."
        ),
    },
    {
        "title": "Product Manager – GenAI for Developers",
        "company": "Amazon Web Services",
        "location": "Seattle, WA",
        "description": (
            "Drive the product roadmap for Amazon Bedrock, AWS's managed service for "
            "building generative AI applications on foundation models from Anthropic, "
            "AI21, Cohere, Meta, and Amazon Titan. You'll define the developer experience "
            "for model access, fine-tuning, knowledge bases (RAG), agents, and guardrails. "
            "5+ years PM experience; cloud services, developer tools, or API platform "
            "background required. Strong technical communication skills — you'll regularly "
            "write product briefs and present to AWS leadership. Partners include AWS AI "
            "services, the Bedrock engineering team, and the AWS go-to-market team. "
            "You'll represent developer needs in model selection decisions and drive "
            "adoption metrics across enterprise accounts."
        ),
    },
    {
        "title": "Senior PM – Responsible AI",
        "company": "IBM",
        "location": "New York, NY (Hybrid)",
        "description": (
            "Lead IBM's responsible AI product strategy for watsonx.governance — an "
            "enterprise platform for AI model monitoring, bias detection, explainability, "
            "and regulatory compliance. You'll work with enterprise risk officers, compliance "
            "teams, and data scientists to ensure AI systems meet EU AI Act, US EO on AI, "
            "and internal governance requirements. Deep understanding of fairness metrics, "
            "model explainability techniques (SHAP, LIME), and AI regulatory landscape is "
            "required. 6+ years PM experience; AI governance, enterprise risk, or regulatory "
            "compliance technology background strongly preferred. You'll publish thought "
            "leadership on responsible AI and represent IBM at industry standards bodies."
        ),
    },
    {
        "title": "Product Manager – AI Agents",
        "company": "LangChain",
        "location": "Remote",
        "description": (
            "Define the product vision for LangSmith, LangChain's observability and "
            "evaluation platform for LLM applications, and LangGraph, our framework for "
            "building multi-agent workflows. Your customers are developers building "
            "production RAG pipelines, AI agents, and LLM-powered applications. "
            "You'll own the roadmap for tracing, evaluation datasets, prompt management, "
            "and human-in-the-loop review flows. Deep familiarity with LLM application "
            "architecture — chains, agents, tool use, memory — is required. 3+ years PM "
            "experience; developer tools or OSS product background preferred. This is a "
            "high-autonomy role in a fast-moving team. You'll engage directly with the "
            "developer community on GitHub and Discord to shape the product."
        ),
    },
    {
        "title": "Senior PM – AI-Powered HR",
        "company": "Workday",
        "location": "Pleasanton, CA",
        "description": (
            "Own the AI features roadmap for Workday's Human Capital Management suite — "
            "including AI-powered skills inference, job architecture, and talent marketplace "
            "recommendations. You'll define how Workday uses NLP and ML to extract employee "
            "skills from job descriptions, performance reviews, and project histories, then "
            "match them to internal opportunities. 4+ years PM experience; HR tech, "
            "enterprise SaaS, or talent management background preferred. Partners include "
            "the Workday AI Lab, product marketing, and enterprise customer success. "
            "Understanding of HR workflows (performance management, succession planning, "
            "workforce planning) is essential. You'll also navigate customer privacy "
            "requirements around employee data used for AI training."
        ),
    },
    {
        "title": "Senior PM – AI Legal Tools",
        "company": "Harvey AI",
        "location": "San Francisco, CA",
        "description": (
            "Build the next generation of AI legal tools at Harvey, where Claude and GPT-4 "
            "help lawyers at elite firms draft contracts, conduct due diligence, and research "
            "case law. You'll own the product roadmap for contract analysis, legal memo "
            "generation, and cross-jurisdictional research features. This role requires "
            "understanding both the capabilities and limitations of LLMs in high-stakes "
            "professional contexts where hallucinations have serious consequences. 4+ years "
            "PM experience; legal tech, enterprise B2B SaaS, or professional services "
            "software background preferred. JD or prior experience at a law firm is a "
            "strong plus. Partners include legal research teams, client success, and the "
            "Harvey AI safety team."
        ),
    },
    {
        "title": "Product Manager – AI Design Tools",
        "company": "Canva",
        "location": "Sydney, Australia (or Remote)",
        "description": (
            "Drive the roadmap for Canva's AI suite — Magic Write, Magic Design, Magic "
            "Eraser, and Dream Lab — which help 170M+ users create professional designs "
            "without design expertise. You'll work with multimodal AI teams to improve "
            "text-to-image quality, expand language support for AI writing, and develop "
            "new AI-powered automation features for template personalization. Strong "
            "consumer product instincts and sensitivity to design quality are essential. "
            "4+ years PM experience; B2C creative tools, media, or consumer AI products "
            "background preferred. You'll partner with ML research, design, and the "
            "localization team to ship AI features loved by users globally."
        ),
    },
    {
        "title": "Senior PM – Agentic AI Products",
        "company": "Cognition",
        "location": "San Francisco, CA",
        "description": (
            "Define the product experience for Devin, the world's first AI software engineer "
            "that can autonomously plan, execute, and iterate on complex coding tasks. "
            "You'll work with the research team to understand the frontier of AI agent "
            "capabilities and translate them into reliable product experiences that software "
            "engineering teams can trust. Core challenges: defining the human-in-the-loop "
            "experience for agent supervision, building evaluation frameworks for task "
            "completion quality, and managing the product lifecycle for an agent that "
            "operates over long horizons. 4+ years PM experience; developer tools or "
            "AI research background required. You must deeply understand software "
            "engineering workflows and be comfortable with ambiguity at the frontier of AI."
        ),
    },
    {
        "title": "Product Manager – AI Education",
        "company": "Khan Academy",
        "location": "Mountain View, CA",
        "description": (
            "Own the product roadmap for Khanmigo, Khan Academy's AI tutor powered by "
            "GPT-4, which guides 130M learners through math, science, and SAT prep without "
            "giving away answers. You'll define pedagogical AI interaction patterns, manage "
            "safety guardrails for a student audience (including minors), and work with "
            "learning scientists to measure educational effectiveness. Strong mission "
            "alignment with education equity is essential. 4+ years PM experience; "
            "ed-tech, consumer AI, or learning science background preferred. Partners "
            "include the AI safety team, curriculum designers, and school district "
            "administrators. Success metrics: learning outcomes, concept mastery rates, "
            "and engagement without screen time concerns."
        ),
    },
    {
        "title": "Senior Product Manager – AI Productivity",
        "company": "Slack (Salesforce)",
        "location": "San Francisco, CA",
        "description": (
            "Lead the product vision for Slack AI — our intelligent assistant that "
            "summarizes channel conversations, answers questions from message history, "
            "and drafts replies in your voice. You'll own the full roadmap from search "
            "quality to AI-generated meeting recaps and action item extraction. 5+ years "
            "PM experience; collaboration software, enterprise SaaS, or conversational "
            "AI background preferred. You'll work with the LLM and infrastructure teams "
            "on retrieval quality, latency optimization, and data privacy for enterprise "
            "customers. Partners include Salesforce Einstein AI, the Slack growth team, "
            "and enterprise sales. Defining evaluation frameworks for summarization quality "
            "and user trust in AI outputs is a core part of the role."
        ),
    },
    {
        "title": "Product Manager – ML Platform",
        "company": "LinkedIn",
        "location": "Sunnyvale, CA",
        "description": (
            "Own the internal ML infrastructure roadmap at LinkedIn — the platform that "
            "powers job recommendations, People You May Know, and feed ranking for 950M "
            "members. You'll manage the roadmap for feature engineering, model training "
            "orchestration, online inference serving, and A/B testing infrastructure. "
            "Strong understanding of large-scale distributed systems and the ML lifecycle "
            "is required. 5+ years PM experience; ML platform, developer tools, or "
            "infrastructure background strongly preferred. Partners include applied ML, "
            "data engineering, and the product teams that consume the platform. "
            "Success metrics include experiment velocity, model deployment frequency, "
            "and platform reliability (SLA for inference serving)."
        ),
    },
    {
        "title": "Senior PM – AI-First Design Features",
        "company": "Figma",
        "location": "San Francisco, CA",
        "description": (
            "Define how AI transforms the design process in Figma — from auto-layout "
            "suggestions to AI-generated components, design-to-code translation, and "
            "intelligent design system governance. You'll work with ML engineers and "
            "designers to ship features that accelerate design workflows without "
            "undermining creative control. 5+ years PM experience; design tools, "
            "creative software, or developer tools background required. Strong design "
            "aesthetic and deep empathy for professional designers' workflows is essential. "
            "Partners include the Figma AI team, plugin ecosystem, and the enterprise "
            "product team. You'll navigate the tension between AI automation and "
            "designer agency throughout the roadmap."
        ),
    },
    {
        "title": "Product Manager – Intelligent Automation",
        "company": "ServiceNow",
        "location": "Santa Clara, CA",
        "description": (
            "Drive the roadmap for ServiceNow's Now Assist AI features — generative AI "
            "embedded across IT, HR, and customer service workflows. Use cases include "
            "AI-generated incident summaries, intelligent ticket routing, case deflection "
            "via virtual agents, and predictive change management. 4+ years PM experience; "
            "enterprise software, ITSM, or workflow automation background preferred. "
            "Understanding of agentic AI patterns and how to build reliable AI automation "
            "for enterprise process management is a plus. Partners include the Now Platform "
            "team, domain SMEs in IT and HR, and the enterprise customer advisory board. "
            "Success metrics: case deflection rate, time-to-resolution, and AI adoption "
            "across enterprise accounts."
        ),
    },
    {
        "title": "Product Manager – AI-Powered Finance",
        "company": "Intuit",
        "location": "Mountain View, CA",
        "description": (
            "Own the AI assistant roadmap for TurboTax and QuickBooks — helping 100M+ "
            "customers navigate taxes and small business finances with conversational AI. "
            "Core features: Intuit Assist (natural language tax Q&A), AI-powered "
            "transaction categorization, cash flow prediction, and automated bookkeeping. "
            "4+ years PM experience; fintech, tax software, or consumer AI products "
            "background preferred. Comfort with regulatory constraints (IRS compliance, "
            "SOC2, financial advice liability) is essential — you'll work with legal and "
            "compliance teams to define the safe boundaries of AI financial advice. "
            "Partners include the Intuit AI Research team, privacy engineering, and "
            "the TurboTax consumer product team."
        ),
    },
    {
        "title": "Senior PM – Computer Vision Platform",
        "company": "Tesla",
        "location": "Palo Alto, CA",
        "description": (
            "Lead the product roadmap for Tesla's vision-based autonomous driving "
            "perception stack — the system that identifies road features, vehicles, "
            "pedestrians, and signals using only cameras (no radar or lidar). You'll "
            "define evaluation criteria for neural network performance, manage the "
            "roadmap for Dojo (Tesla's custom AI training supercomputer), and own the "
            "product lifecycle for AutoPilot features from training data requirements "
            "to fleet deployment. 5+ years PM experience; computer vision, embedded "
            "systems, or automotive software background required. Strong engineering "
            "collaboration skills and comfort in a high-pressure, mission-critical "
            "environment are non-negotiable. The safety implications of this role are "
            "among the highest in the industry."
        ),
    },
    {
        "title": "Product Manager – AI Ops",
        "company": "PagerDuty",
        "location": "San Francisco, CA",
        "description": (
            "Own the product roadmap for PagerDuty's AIOps capabilities — intelligent "
            "noise reduction, automated incident triage, and ML-powered on-call "
            "recommendations for site reliability engineers. You'll work with data "
            "scientists building anomaly detection, event correlation, and time-series "
            "forecasting models that help SRE teams respond to incidents faster. "
            "4+ years PM experience; DevOps, SRE tools, or observability background "
            "preferred. Understanding of incident management workflows and the SRE "
            "lifecycle is essential. Partners include the AI team, enterprise customer "
            "success, and the platform engineering team. Success metrics: mean time to "
            "detect (MTTD), noise reduction ratio, and on-call burnout indicators."
        ),
    },
    {
        "title": "Senior PM – AI-Powered Insights",
        "company": "Mixpanel",
        "location": "San Francisco, CA",
        "description": (
            "Define how AI transforms product analytics at Mixpanel — from natural "
            "language querying of event data to AI-generated insight summaries and "
            "anomaly alerts. You'll own the roadmap for Mixpanel's AI assistant that "
            "lets PMs ask 'Why did DAU drop last Tuesday?' in plain English and get "
            "a grounded answer with supporting charts. 4+ years PM experience; product "
            "analytics, business intelligence, or data products background preferred. "
            "Strong analytical skills and a deep understanding of how growth teams "
            "use data is essential. Partners include the data science team, customer "
            "success, and the core analytics platform team. This role will define the "
            "future of self-serve product analytics with AI assistance."
        ),
    },
    {
        "title": "Senior PM – Knowledge Management AI",
        "company": "Atlassian (Confluence)",
        "location": "Austin, TX (or Remote)",
        "description": (
            "Build the AI roadmap for Confluence, Atlassian's team wiki used by 50,000+ "
            "companies. AI initiatives include Atlassian Intelligence for page summarization, "
            "smart search across Confluence and Jira, AI-generated content suggestions, "
            "and automated knowledge graph extraction. 5+ years PM experience; "
            "enterprise collaboration tools, knowledge management, or content platforms "
            "background preferred. You'll partner with the Atlassian AI team, Jira PM "
            "counterparts, and large enterprise customers. Understanding of how engineering "
            "and product teams use documentation tools — and their pain points around "
            "knowledge discovery — is essential. You'll define quality metrics for AI "
            "search relevance and summarization accuracy."
        ),
    },
    {
        "title": "Product Manager – Conversational UI",
        "company": "Intercom",
        "location": "San Francisco, CA",
        "description": (
            "Own the roadmap for Intercom's Fin AI Agent — an LLM-powered customer support "
            "bot that resolves 50%+ of support tickets automatically. You'll define how Fin "
            "uses RAG over customer help centers, escalates gracefully to human agents, and "
            "learns from conversation history. Core focus: answer accuracy, resolution rate, "
            "and customer satisfaction (CSAT). 4+ years PM experience; customer support "
            "software, conversational AI, or B2B SaaS background preferred. Deep "
            "understanding of support workflows and the metrics that matter to customer "
            "success leaders (CSAT, FCR, AHT) is essential. Partners include the AI "
            "research team, customer success, and the Intercom platform engineering team."
        ),
    },
    {
        "title": "Senior PM – Multimodal AI",
        "company": "Google (Gemini Team)",
        "location": "Mountain View, CA",
        "description": (
            "Define the product experience for Gemini's multimodal capabilities — "
            "understanding and generating across text, images, audio, and video. "
            "You'll work with research teams pushing the frontier of vision-language "
            "models and translate capabilities into compelling user experiences for "
            "Google Search, Workspace, and the Gemini app. 5+ years PM experience; "
            "multimodal AI, computer vision, or consumer AI products background preferred. "
            "Strong ability to evaluate model quality through human evaluation and "
            "automated benchmarks is essential. Partners include Google Research, "
            "Search, YouTube, and Google Cloud. You'll navigate the tradeoffs between "
            "capability, safety, and latency in one of the most competitive AI product "
            "spaces in the world."
        ),
    },
    {
        "title": "Product Manager – AI Code Generation",
        "company": "JetBrains",
        "location": "Remote (Europe/Americas)",
        "description": (
            "Build the product roadmap for AI Assistant in JetBrains IDEs — code "
            "completion, explanation, refactoring suggestions, and test generation "
            "for 12M+ developers. You'll work with LLM teams on context management "
            "strategies (how much of the codebase to include in the prompt), local "
            "vs cloud inference tradeoffs, and language-specific quality benchmarks. "
            "4+ years PM experience; developer tools, IDE products, or programming "
            "language ecosystem background required. You should be comfortable reading "
            "code and have strong opinions about what makes developer AI tools excellent "
            "vs annoying. Partners include the IntelliJ IDEA, PyCharm, and WebStorm "
            "product teams."
        ),
    },
    {
        "title": "Senior PM – AI-Powered Customer Success",
        "company": "Zendesk",
        "location": "San Francisco, CA",
        "description": (
            "Lead the product roadmap for Zendesk's AI suite — automated ticket "
            "classification, intelligent agent assist (surface relevant articles during "
            "live chats), and AI-generated CSAT prediction. You'll own the experience "
            "for both support agents (who need AI that makes their job easier) and "
            "customers (who want fast, accurate answers). 4+ years PM experience; "
            "customer service software, CRM, or B2B SaaS background preferred. "
            "Understanding of support center operations — ticket volume, agent "
            "productivity, SLA management — is essential. Partners include Zendesk "
            "AI, the marketplace/apps team, and enterprise customer advisory boards. "
            "You'll define evaluation frameworks for AI accuracy that customers "
            "can trust and audit."
        ),
    },
    {
        "title": "AI Product Manager – OpenAI Platform",
        "company": "OpenAI",
        "location": "San Francisco, CA",
        "description": (
            "Drive the developer platform roadmap for OpenAI API — the API used by "
            "2M+ developers to build on GPT-4, DALL-E, Whisper, and Embeddings. "
            "You'll own the roadmap for Assistants API (stateful agents with tool "
            "use, code interpreter, and retrieval), the fine-tuning experience, and "
            "the usage dashboard. Deep understanding of LLM application architecture "
            "and developer needs at every stage (prototype → production) is required. "
            "5+ years PM experience; API platform, developer ecosystem, or AI "
            "infrastructure background preferred. Partners include the research team, "
            "safety, DevRel, and enterprise sales. You'll frequently engage with the "
            "developer community to understand how they are using the API and what "
            "they need next."
        ),
    },
    {
        "title": "Senior PM – AI Document Intelligence",
        "company": "Microsoft Azure AI",
        "location": "Redmond, WA (or Remote)",
        "description": (
            "Own the product roadmap for Azure AI Document Intelligence — OCR, "
            "form recognition, and document understanding APIs used by enterprises "
            "to extract structured data from invoices, contracts, medical records, "
            "and government forms. You'll work with computer vision and NLP teams "
            "to improve extraction accuracy across document types and languages. "
            "4+ years PM experience; document AI, enterprise APIs, or B2B cloud "
            "services background preferred. Understanding of enterprise document "
            "processing workflows (accounts payable, contract lifecycle management, "
            "compliance) is a plus. Partners include Azure AI Services, the SDK team, "
            "and enterprise field engineering. Success metrics: extraction accuracy, "
            "API adoption, and customer satisfaction scores."
        ),
    },
    {
        "title": "Senior Product Manager – AI Strategy",
        "company": "McKinsey & Company (QuantumBlack)",
        "location": "New York, NY",
        "description": (
            "Lead AI product development at QuantumBlack, McKinsey's AI subsidiary, "
            "building proprietary AI products and platforms used by global clients "
            "across industries. You'll manage the roadmap for internal AI tools that "
            "help consultants synthesize research, generate client presentations, and "
            "identify strategic opportunities using large-scale data analysis. "
            "6+ years of experience combining strategy consulting and product management "
            "required. Comfort with AI/ML concepts at a strategic level (not necessarily "
            "implementation-level) is essential. MBA from a top institution preferred. "
            "Partners include data scientists, engagement managers, and senior "
            "partners across McKinsey practices. This is a high-impact role at the "
            "intersection of management consulting and AI product development."
        ),
    },
    {
        "title": "Senior PM – Computer Vision AI",
        "company": "Snap",
        "location": "Santa Monica, CA",
        "description": (
            "Drive the product roadmap for Snap's AR and computer vision features — "
            "including real-time face tracking, body segmentation, 3D scene understanding, "
            "and generative AR effects. You'll work with the Lens Studio platform team "
            "and computer vision researchers to define the next generation of creative "
            "tools for Snapchat's 800M+ user community. 5+ years PM experience; AR/VR, "
            "computer vision, or mobile consumer AI background preferred. Strong "
            "creative instincts and understanding of Gen Z digital culture is a plus. "
            "Partners include the AR platform team, content partnerships, and "
            "third-party Lens creators. You'll navigate hardware constraints (mobile "
            "inference) while pushing the frontier of what's possible on device."
        ),
    },
    {
        "title": "Product Manager – AI Trust & Safety",
        "company": "Meta",
        "location": "Menlo Park, CA",
        "description": (
            "Build AI-powered content moderation and trust infrastructure for Meta's "
            "family of apps — Facebook, Instagram, WhatsApp, and Threads. You'll "
            "define the product roadmap for integrity classifiers that detect hate "
            "speech, misinformation, spam, and synthetic media at billions-per-day "
            "scale. Understanding of precision/recall tradeoffs, adversarial robustness, "
            "and the policy implications of automated moderation decisions is essential. "
            "5+ years PM experience; trust & safety, content moderation, or policy "
            "technology background required. Partners include the AI integrity research "
            "team, policy, legal, and communications. This role has significant "
            "societal impact and high media visibility — comfort with ambiguity and "
            "nuanced ethical tradeoffs is non-negotiable."
        ),
    },
    {
        "title": "Senior PM – AI Infrastructure",
        "company": "Cloudflare",
        "location": "San Francisco, CA (or Remote)",
        "description": (
            "Define the product roadmap for Cloudflare Workers AI — serverless "
            "inference at the edge, letting developers run LLMs and image models "
            "on Cloudflare's global network with ultra-low latency. You'll work "
            "with the network and ML engineering teams to expand model availability, "
            "optimize inference performance (tokens/second), and build developer "
            "experience for AI-powered Cloudflare Workers and Pages applications. "
            "4+ years PM experience; cloud infrastructure, developer platform, or "
            "edge computing background preferred. Understanding of inference "
            "optimization (quantization, batching, speculative decoding) and "
            "developer needs for AI APIs is a plus. Partners include the Workers "
            "platform team, the model partnerships team, and enterprise sales."
        ),
    },
    {
        "title": "Product Manager – AI Health Diagnostics",
        "company": "Tempus AI",
        "location": "Chicago, IL",
        "description": (
            "Own the product roadmap for Tempus's AI clinical decision support tools "
            "that help oncologists identify optimal treatments using genomic data, "
            "clinical records, and imaging results. You'll work with data scientists "
            "building predictive models for treatment response and collaborate with "
            "clinical partners to validate AI output in real patient care settings. "
            "4+ years PM experience; digital health, clinical informatics, or "
            "life sciences background strongly preferred. Familiarity with FDA "
            "regulatory pathways for AI-based Software as a Medical Device (SaMD) "
            "is a significant plus. Partners include clinical operations, "
            "bioinformatics, and hospital system integration teams. Mission: "
            "make precision medicine accessible to every patient."
        ),
    },
    {
        "title": "Senior PM – AI Supply Chain",
        "company": "Amazon (Supply Chain Optimization)",
        "location": "Seattle, WA",
        "description": (
            "Lead the AI product roadmap for Amazon's supply chain optimization "
            "systems — demand forecasting, inventory placement, transportation "
            "routing, and warehouse automation. You'll work with operations research "
            "scientists and ML engineers to improve models that move billions of "
            "packages through Amazon's global fulfillment network. 5+ years PM "
            "experience; supply chain, logistics, or operations research background "
            "preferred. Strong quantitative skills and comfort with optimization "
            "concepts (linear programming, simulation, reinforcement learning for "
            "operations) are a plus. Partners include fulfillment center operations, "
            "transportation, and the supply chain platform engineering team. Success "
            "metrics: inventory turns, in-stock rate, and cost-per-unit-shipped."
        ),
    },
    {
        "title": "Director of Product – AI Platform",
        "company": "Palantir",
        "location": "New York, NY or Denver, CO",
        "description": (
            "Lead the AI Platform product team at Palantir, building AIP (Artificial "
            "Intelligence Platform) — the enterprise product that helps government and "
            "commercial customers deploy and orchestrate LLMs safely on their own data. "
            "You'll manage a team of PMs and define the strategy for LLM integration, "
            "ontology-grounded retrieval, and human-in-the-loop workflows across "
            "defense, intelligence, and commercial verticals. 7+ years PM experience; "
            "2+ years managing PMs; government or national security background is "
            "a plus. Ability to obtain a security clearance may be required. Partners "
            "include Palantir's forward-deployed engineers (FDEs), enterprise field "
            "teams, and government agency stakeholders. You'll operate at the "
            "intersection of AI capability and mission-critical enterprise use cases."
        ),
    },
    {
        "title": "Product Manager – Generative Media AI",
        "company": "Runway ML",
        "location": "New York, NY",
        "description": (
            "Define the product experience for Runway Gen-3 — our state-of-the-art "
            "video generation model used by filmmakers, advertising agencies, and "
            "creative studios worldwide. You'll work at the frontier of multimodal "
            "AI, translating research breakthroughs in video generation into "
            "intuitive creative tools. Responsibilities include defining evaluation "
            "criteria for video quality, managing the beta rollout of new generation "
            "capabilities, and building creative community feedback loops. 4+ years "
            "PM experience; creative software, media production, or generative AI "
            "background strongly preferred. Strong aesthetic sensibility and "
            "understanding of professional creative workflows is essential. This is "
            "a high-visibility role at one of the most exciting AI companies in the "
            "creative space."
        ),
    },
    {
        "title": "Senior PM – AI Financial Research",
        "company": "Bloomberg",
        "location": "New York, NY",
        "description": (
            "Own the product roadmap for Bloomberg's AI research assistant — an "
            "LLM-powered tool that helps financial analysts synthesize earnings "
            "calls, SEC filings, analyst reports, and market data into actionable "
            "investment insights. You'll work with the Bloomberg Intelligence team "
            "and ML engineers to improve retrieval quality, financial entity "
            "recognition, and numerical reasoning in LLM outputs. 5+ years PM "
            "experience; fintech, financial data, or enterprise financial software "
            "background required. CFA or prior buy-side/sell-side experience is "
            "a significant plus. Understanding of regulatory constraints around "
            "AI-generated financial analysis is essential. Partners include the "
            "Terminal product team, Bloomberg Intelligence analysts, and "
            "enterprise client teams."
        ),
    },
    {
        "title": "Product Manager – AI Manufacturing",
        "company": "Siemens",
        "location": "Munich, Germany (or Remote Europe)",
        "description": (
            "Build AI product features for Siemens Industrial Copilot — generative "
            "AI embedded in industrial automation software to help factory engineers "
            "diagnose equipment faults, optimize production schedules, and write "
            "PLC code. You'll translate industrial engineering workflows into "
            "AI product requirements and work with ML engineers specializing in "
            "time-series anomaly detection, digital twin simulation, and predictive "
            "maintenance. 4+ years PM experience; industrial software, manufacturing "
            "tech, or B2B enterprise background preferred. German language skills "
            "are a plus but not required. Partners include Siemens Healthineers, "
            "the Digital Industries division, and global manufacturing customers. "
            "This role is at the frontier of applying LLMs in operational technology (OT)."
        ),
    },
]


def embed(text: str, openai_client: OpenAI) -> list[float]:
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000],
    )
    return resp.data[0].embedding


def main() -> None:
    missing = [
        k for k in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY", "OPENAI_API_KEY", "SEED_USER_ID")
        if not os.getenv(k)
    ]
    if missing:
        print("ERROR: Missing environment variables:", ", ".join(missing))
        print("\nAdd these to your .env file:")
        print("  SUPABASE_SERVICE_KEY — Dashboard → Settings → API → service_role key")
        print("  SEED_USER_ID         — Dashboard → Authentication → Users → your UUID")
        sys.exit(1)

    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"Seeding {len(JOBS)} job postings into Supabase...\n")

    for i, job in enumerate(JOBS, 1):
        print(f"[{i:2d}/{len(JOBS)}] {job['title']} @ {job['company']}")
        try:
            embedding = embed(job["description"], openai_client)
            supabase.table("job_postings").insert({
                "user_id": SEED_USER_ID,
                "title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "description": job["description"],
                "embedding": embedding,
            }).execute()
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        # Small pause to stay within OpenAI rate limits (500 RPM for embeddings)
        if i % 10 == 0:
            time.sleep(1)

    print(f"\nDone. {len(JOBS)} postings seeded.")
    print("Open the Job Matcher page and click 'Find My Best Matches' to test.")


if __name__ == "__main__":
    main()
