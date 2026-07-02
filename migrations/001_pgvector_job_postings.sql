-- Phase 1: pgvector + job_postings table + cosine similarity RPC
-- Run this in: Supabase Dashboard → SQL Editor → New query → Run

-- ── Step 1: Enable pgvector ───────────────────────────────────────────────────
-- pgvector is a Postgres extension that adds a "vector" column type and
-- operators like <=> (cosine distance) for nearest-neighbor search.
-- Supabase projects have it available but it must be explicitly enabled.
create extension if not exists vector;


-- ── Step 2: Create the job_postings table ────────────────────────────────────
-- Each row = one job posting + its 1536-dimensional embedding.
-- The embedding column is what pgvector indexes and searches.
create table if not exists job_postings (
  id          uuid        default gen_random_uuid() primary key,
  user_id     uuid        references auth.users(id) not null,
  title       text        not null default '',
  company     text        not null default '',
  location    text        not null default '',
  description text        not null,
  embedding   vector(1536),   -- OpenAI text-embedding-3-small produces 1536 floats
  created_at  timestamptz default now()
);


-- ── Step 3: Row-level security ────────────────────────────────────────────────
-- Supabase enforces RLS by default. These policies say:
--   - Anyone logged in can READ all postings (shared job directory)
--   - You can only INSERT/DELETE your own rows
alter table job_postings enable row level security;

create policy "Authenticated users can read all postings"
  on job_postings for select
  to authenticated
  using (true);

create policy "Users insert their own postings"
  on job_postings for insert
  to authenticated
  with check (auth.uid() = user_id);

create policy "Users delete their own postings"
  on job_postings for delete
  to authenticated
  using (auth.uid() = user_id);

-- Allow the service role (seed script) to bypass RLS
create policy "Service role full access"
  on job_postings
  to service_role
  using (true)
  with check (true);


-- ── Step 4: IVFFlat index for fast vector search ──────────────────────────────
-- Without an index, finding similar vectors requires comparing every row
-- (exact search, O(n)). IVFFlat is an approximate nearest-neighbor index that
-- clusters vectors into "lists" and only searches nearby clusters.
--
-- lists = 10 is correct for up to ~1000 rows.
-- At 100k+ rows, increase lists to 100 and run: SET ivfflat.probes = 10;
create index if not exists job_postings_embedding_idx
  on job_postings using ivfflat (embedding vector_cosine_ops)
  with (lists = 10);


-- ── Step 5: Cosine similarity RPC ────────────────────────────────────────────
-- This is the function storage.py calls as: supabase.rpc("match_job_postings", ...)
-- It accepts a query vector (your resume embedding) and returns the top-k
-- most similar job postings, with their cosine similarity score.
--
-- The <=> operator computes cosine DISTANCE (0=identical, 2=opposite).
-- We convert to SIMILARITY with: 1 - distance (so 1=identical, -1=opposite).
--
-- match_user_id is accepted for backward compatibility but not used in the
-- WHERE clause — the shared directory searches across all users.
create or replace function match_job_postings(
  query_embedding vector(1536),
  match_user_id   uuid,
  match_count     int default 10
)
returns table (
  id          uuid,
  user_id     uuid,
  title       text,
  company     text,
  location    text,
  description text,
  similarity  float
)
language sql stable
as $$
  select
    id,
    user_id,
    title,
    company,
    location,
    description,
    1 - (embedding <=> query_embedding) as similarity
  from job_postings
  where embedding is not null
  order by embedding <=> query_embedding
  limit match_count;
$$;


-- ── Step 6: pkce_verifiers table (auth, if not already created) ───────────────
-- Required by auth.py to persist PKCE code verifiers across Streamlit workers.
-- Safe to run even if it already exists (the IF NOT EXISTS guard).
create table if not exists pkce_verifiers (
  token     text primary key,
  verifier  text not null,
  created_at timestamptz default now()
);

-- Auto-delete verifiers older than 10 minutes (they're single-use)
create index if not exists pkce_verifiers_created_at_idx
  on pkce_verifiers (created_at);
