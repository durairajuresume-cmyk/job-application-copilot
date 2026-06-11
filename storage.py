from supabase import Client


# ── Resume ────────────────────────────────────────────────────────────────────

def save_resume(
    supabase: Client,
    user_id: str,
    file_bytes: bytes,
    file_name: str,
    resume_text: str,
    resume_json: dict = None,
) -> None:
    """Upload PDF to Storage (best-effort) and upsert the metadata row."""
    path = f"{user_id}/resume.pdf"
    try:
        supabase.storage.from_("resumes").upload(
            path,
            file_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"},
        )
    except Exception:
        path = ""  # bucket missing or upload failed — still save the text

    # Always save the text/metadata row so the resume is retrievable on next login
    supabase.table("user_resumes").upsert(
        {
            "user_id": user_id,
            "file_name": file_name,
            "file_path": path,
            "resume_text": resume_text,
            "resume_json": resume_json or {},
        },
        on_conflict="user_id",
    ).execute()


def load_resume_meta(supabase: Client, user_id: str) -> dict | None:
    """Return the saved resume metadata row, or None."""
    try:
        resp = (
            supabase.table("user_resumes")
            .select("*")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        return rows[0] if rows else None
    except Exception:
        return None


def download_resume_pdf(supabase: Client, user_id: str) -> bytes | None:
    """Download the stored PDF bytes, or None if not found."""
    try:
        return bytes(
            supabase.storage.from_("resumes").download(f"{user_id}/resume.pdf")
        )
    except Exception:
        return None


# ── Application history ───────────────────────────────────────────────────────

def save_application(supabase: Client, user_id: str, payload: dict) -> None:
    supabase.table("applications").insert({"user_id": user_id, **payload}).execute()


def list_applications(supabase: Client, user_id: str) -> list:
    resp = (
        supabase.table("applications")
        .select("id, job_title, company, match_score, status, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []
