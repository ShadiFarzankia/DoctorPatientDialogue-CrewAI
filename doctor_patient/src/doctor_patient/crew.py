# src/doctor_patient/crew.py
from __future__ import annotations

from typing import List
import requests
import json
import re

from .tools.retrieval import (
    get_dialogues_and_raw_for_chief_complaint,  # subjective-based retrieval
    get_candidate_drugs_for_symptoms,           # for the drug flow
    get_similar_cases_for_summary,              # for the summary flow
)

# Ollama HTTP endpoint
OLLAMA_URL = "http://localhost:11434/api/chat"


# ---------------------------------------------------------------------
# Low-level Ollama helper
# ---------------------------------------------------------------------
def ollama_chat(prompt: str, model: str = "llama3") -> str:
    """Send a chat completion request directly to Ollama."""
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": 0.4,
        }

        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Ollama's /api/chat usually returns: {"message": {"role": "...", "content": "..."}}
        return data.get("message", {}).get("content", "") or ""
    except Exception as e:
        print("[ollama error]", e)
        return ""


# ---------------------------------------------------------------------
# Helper: pull JSON out of messy LLM output
# ---------------------------------------------------------------------
def _extract_json_dict(raw: str) -> dict | None:
    """Try to pull a JSON object out of an LLM response that may contain text + code blocks."""
    raw = (raw or "").strip()
    if not raw:
        return None

    # 1) Maybe it's already pure JSON
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2) Look for ```json ...``` or ``` ...``` fenced block
    m = re.search(r"```(?:json)?\s*({.*?})\s*```", raw, flags=re.DOTALL)
    if m:
        candidate = m.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # 3) Fallback: first {...} anywhere
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------
# SYMPTOM FLOW
# ---------------------------------------------------------------------
def run_symptom_flow(chief_complaint: str) -> List[str]:
    """
    Given a free-text chief complaint, use train_subjective.json via
    get_dialogues_and_raw_for_chief_complaint to find similar dialogues,
    then ask the LLM to propose symptom phrases.

    Returns a list of strings suitable for checkboxes.
    """
    chief_complaint = (chief_complaint or "").strip()
    if not chief_complaint:
        return []

    # 1) Get top similar dialogues from train_subjective.json
    top_dialogues = get_dialogues_and_raw_for_chief_complaint(
        chief_complaint,
        k=2,  # top 2 as you requested
    )

    # Build a compact context for the LLM
    dialog_snippets: List[str] = []
    for i, d in enumerate(top_dialogues, start=1):
        # Depending on your retrieval implementation, you may have keys like
        # "dialogue" or "src" or "raw"
        text = (
            d.get("dialogue")
            or d.get("src")
            or d.get("raw", {}).get("src")
            or ""
        )
        text = str(text)
        if len(text) > 800:
            text = text[:800] + "..."
        dialog_snippets.append(f"--- DIALOGUE {i} ---\n{text}")

    context_block = "\n\n".join(dialog_snippets) if dialog_snippets else "[none found]"

    # 2) Prompt LLM to suggest co-occurring symptoms
    prompt = f"""
You are a clinical symptom extraction helper for a research-only prototype.
You are given a patient's chief complaint and a few similar historical
doctor–patient dialogues (from ACI-Bench).

Patient chief complaint:
\"\"\"{chief_complaint}\"\"\"

Similar historical dialogues:
{context_block}

Your task:

1. Infer up to 5 short symptom phrases that could *reasonably co-occur*
   with this patient's complaint, based on the patterns you see in the
   similar dialogues.
2. Each phrase must be concise (<= 12 words).
3. Do not mention diagnoses or treatments, only *symptoms* or *feelings*.
4. Return ONLY a JSON object with this exact structure:

{{
  "symptom_options": [
    "symptom phrase 1",
    "symptom phrase 2"
  ]
}}

You MAY include some explanation in natural language BEFORE the JSON,
but the JSON block itself must be valid.
"""

    raw = ollama_chat(prompt)
    data = _extract_json_dict(raw)

    if data is None:
        print("[symptom_flow] Failed to parse JSON:", repr(raw)[:300])

        return []

    opts = data.get("symptom_options") or []
    if not isinstance(opts, list):
        print(f"[symptom_flow] symptom_options is not a list: {data!r}")
        return []

    cleaned: List[str] = []
    for o in opts:
        if isinstance(o, str):
            o = o.strip()
            if o and o not in cleaned:
                cleaned.append(o)

    return cleaned[:5]


# ---------------------------------------------------------------------
# DRUG FLOW  (still using existing retrieval)
# ---------------------------------------------------------------------
def run_drug_flow(chief_complaint: str, selected_symptoms: List[str]) -> List[str]:
    """
    Given confirmed symptoms, return candidate drug names using your
    existing retrieval logic (likely from train_full.json).
    """
    selected_symptoms = selected_symptoms or []
    if not selected_symptoms:
        return []

    candidates = get_candidate_drugs_for_symptoms(selected_symptoms)

    seen: List[str] = []
    for c in candidates:
        name = c.get("name")
        if isinstance(name, str):
            name = name.strip()
            if name and name not in seen:
                seen.append(name)

    return seen[:5]


# ---------------------------------------------------------------------
# SUMMARY FLOW  (SOAP-style, using Ollama)
# ---------------------------------------------------------------------
def run_summary_flow(
    chief_complaint: str,
    selected_symptoms: List[str],
    selected_drugs: List[str],
) -> str:
    """
    Generate a SOAP-style summary using similar cases (retrieval) + llama3
    via Ollama. This is still a research-only, non-medical tool.
    """
    chief_complaint = (chief_complaint or "").strip()
    selected_symptoms = selected_symptoms or []
    selected_drugs = selected_drugs or []

    similar_cases = get_similar_cases_for_summary(
        selected_symptoms=selected_symptoms,
        selected_drugs=selected_drugs,
        max_cases=3,
    )

    # keep a compact version for the prompt
    compact = []
    for case in similar_cases:
        compact.append(
            {
                "chief_complaint": case.get("chief_complaint")
                or case.get("chiefComplaint")
                or case.get("subjective", {}).get("chief_complaint")
                or "",
                "symptoms": case.get("symptoms")
                or case.get("subjective", {}).get("symptoms")
                or [],
                "medications": case.get("medications")
                or case.get("drugs")
                or case.get("objective", {}).get("medications")
                or [],
            }
        )

    prompt = f"""
You are a medical documentation assistant (RESEARCH DEMO ONLY — NOT MEDICAL ADVICE).

Write a compact SOAP-style note from the following information.

Chief complaint:
{chief_complaint or "[not provided]"}

Confirmed symptoms:
{selected_symptoms}

Confirmed medications / drug history:
{selected_drugs}

Similar cases (compact JSON view):
{json.dumps(compact, ensure_ascii=False)}

Requirements:

### Subjective
- 1–3 sentences combining chief complaint and key symptoms.

### Objective
- Describe which exams/tests are *typically* done (vital signs, labs, imaging, etc.)
- Do NOT invent specific numeric values.

### Assessment
- Generic, no diagnosis. Example style:
  "Symptoms suggest a possible acute condition, but further evaluation is needed."

### Plan
- Only HIGH-LEVEL actions such as:
  - book appointment with a primary care physician
  - consider physical examination
  - clinician may consider lab tests or imaging
- Do NOT recommend specific drugs or doses.
- Do NOT give emergency / triage advice.
- Do NOT talk directly to the patient; write as a neutral note.
"""

    text = ollama_chat(prompt)
    text = (text or "").strip()

    if not text:
        return (
            "### Subjective\n"
            "_No summary generated (model returned empty response)._  \n\n"
            "### Objective\n"
            "_Not available._\n\n"
            "### Assessment\n"
            "_Not available._\n\n"
            "### Plan\n"
            "_Please contact a qualified clinician for a real medical assessment._"
        )

    return text
