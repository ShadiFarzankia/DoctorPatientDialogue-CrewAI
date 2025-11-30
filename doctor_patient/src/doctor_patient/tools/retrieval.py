from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------
# Paths (adapted to your repo layout)
# ---------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent          # .../src/doctor_patient/tools
PKG_DIR = TOOLS_DIR.parent                           # .../src/doctor_patient
SRC_ROOT = PKG_DIR.parent                            # .../src
DATA_DIR = SRC_ROOT / "data"                         # .../src/data

SUBJ_PATH = DATA_DIR / "train_subjective.json"

# ---------------------------------------------------------------------
# In-memory cache (ONLY subjective cases)
# ---------------------------------------------------------------------

_SUBJ_CASES: List[Dict[str, Any]] | None = None


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_subjective_cases(raw: Any) -> List[Dict[str, Any]]:
    """
    train_subjective.json format:

    { "data": [ { "src": "...", "subjective": "..." }, ... ] }

    We convert each item into:

    {
      "id": int,
      "dialogue": str,      # all subjective/dialogue text
      "raw": dict
    }
    """
    if isinstance(raw, dict) and "data" in raw:
        items = raw["data"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    norm: List[Dict[str, Any]] = []
    for idx, item in enumerate(items):
        dialogue = (item.get("src") or item.get("subjective") or "").strip()
        if not dialogue:
            continue

        norm.append(
            {
                "id": idx,
                "dialogue": dialogue,
                "raw": item,
            }
        )

    return norm


def _load_subjective_cases() -> List[Dict[str, Any]]:
    global _SUBJ_CASES
    if _SUBJ_CASES is not None:
        return _SUBJ_CASES

    raw = _load_json(SUBJ_PATH)
    _SUBJ_CASES = _normalize_subjective_cases(raw)
    print(f"[retrieval] Loaded {len(_SUBJ_CASES)} subjective cases from {SUBJ_PATH}")
    return _SUBJ_CASES


def build_stores() -> None:
    """Load only the subjective dataset."""
    _load_subjective_cases()


# ---------------------------------------------------------------------
# Basic text similarity helpers
# ---------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    text = (text or "").lower().replace("\n", " ")
    out = []
    for t in text.split():
        t = re.sub(r"[^\w\-]", "", t)
        if t:
            out.append(t)
    return out


def _jaccard_similarity(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = sa & sb
    union = sa | sb
    if not union:
        return 0.0
    return len(inter) / len(union)


# ---------------------------------------------------------------------
# Minimal symptom/medication extractors (from subjective text)
# ---------------------------------------------------------------------

def _extract_chief_complaint_text(text: str) -> str:
    if not text:
        return ""

    marker = "CHIEF COMPLAINT"
    up = text.upper()
    idx = up.find(marker)
    if idx == -1:
        return ""
    after = text[idx + len(marker):]
    for line in after.splitlines():
        line = line.strip(" \t\r\n.:")
        if line:
            return line
    return ""


def _extract_symptom_phrases(text: str) -> List[str]:
    """Extract lines containing 'Endorses ...' if present."""
    if not text:
        return []
    out = []
    for line in text.splitlines():
        if "Endorses" in line:
            seg = line.split("Endorses", 1)[1].strip(" .:-")
            if seg and seg not in out:
                out.append(seg)
    return out


_MED_STOPWORDS = {
    "mg", "daily", "Every", "Refill",
    "We", "I", "He", "She", "They",
    "Continue", "Start", "Initiate",
    "Repeat", "Order",
}


def _extract_medications(text: str) -> List[str]:
    if not text:
        return []
    meds = set()
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_-]*)\s+\d+\s*mg", re.IGNORECASE)

    for line in text.splitlines():
        if "mg" not in line:
            continue
        for m in pattern.finditer(line):
            name = m.group(1).strip(".,;:()[]")
            if name and name not in _MED_STOPWORDS:
                meds.add(name)

    return sorted(meds)


# ---------------------------------------------------------------------
# 1) Similar dialogues for chief complaint
# ---------------------------------------------------------------------

def get_dialogues_and_raw_for_chief_complaint(
    chief_complaint: str,
    k: int = 2,
) -> List[Dict[str, Any]]:
    chief_complaint = (chief_complaint or "").strip()
    if not chief_complaint:
        return []

    cases = _load_subjective_cases()
    q_tokens = _tokenize(chief_complaint)

    scored = []
    for c in cases:
        tokens = _tokenize(c["dialogue"])
        score = _jaccard_similarity(q_tokens, tokens)
        if score > 0:
            scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:k]]


# ---------------------------------------------------------------------
# 2) Symptom candidates (subjective only)
# ---------------------------------------------------------------------

def get_candidate_symptoms_for_chief_complaint(
    chief_complaint: str,
    max_cases: int = 5,
) -> List[Dict[str, Any]]:
    chief_complaint = (chief_complaint or "").strip()
    if not chief_complaint:
        return []

    cases = _load_subjective_cases()
    q_tokens = _tokenize(chief_complaint)

    scored = []
    for c in cases:
        tokens = _tokenize(c["dialogue"])
        score = _jaccard_similarity(q_tokens, tokens)
        if score > 0:
            scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = [c for _, c in scored[:max_cases]]

    out = []
    seen = set()

    for c in top:
        txt = c["dialogue"]

        cc = _extract_chief_complaint_text(txt)
        if not cc:
            cc = txt[:80] + "..." if len(txt) > 80 else txt

        name = cc.strip()
        if name and name not in seen:
            seen.add(name)
            out.append({"name": name, "case_id": c["id"]})

    return out


# ---------------------------------------------------------------------
# 3) Drug candidates (subjective only)
# ---------------------------------------------------------------------

def get_candidate_drugs_for_symptoms(
    selected_symptoms: List[str],
    max_cases: int = 10,
) -> List[Dict[str, Any]]:
    selected_symptoms = selected_symptoms or []
    if not selected_symptoms:
        return []

    cases = _load_subjective_cases()
    q_tokens = _tokenize(" ".join(selected_symptoms))

    scored = []
    for c in cases:
        tokens = _tokenize(c["dialogue"])
        score = _jaccard_similarity(q_tokens, tokens)
        if score > 0:
            scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = [c for _, c in scored[:max_cases]]

    out = []
    seen = set()

    for c in top:
        meds = _extract_medications(c["dialogue"])
        for m in meds:
            if m not in seen:
                seen.add(m)
                out.append({"name": m, "case_id": c["id"]})

    return out


# ---------------------------------------------------------------------
# 4) Similar cases for summary (subjective only)
# ---------------------------------------------------------------------

def get_similar_cases_for_summary(
    selected_symptoms: List[str],
    selected_drugs: List[str],
    max_cases: int = 3,
) -> List[Dict[str, Any]]:
    cases = _load_subjective_cases()

    query_bits = (selected_symptoms or []) + (selected_drugs or [])
    q_tokens = _tokenize(" ".join(query_bits))

    # If nothing chosen → return first N subjective cases
    if not query_bits:
        out = []
        for c in cases[:max_cases]:
            txt = c["dialogue"]
            cc = _extract_chief_complaint_text(txt)
            sym = _extract_symptom_phrases(txt)
            meds = _extract_medications(txt)
            out.append(
                {
                    "chief_complaint": cc,
                    "symptoms": sym,
                    "medications": meds,
                    "drugs": meds,
                    "objective": {"medications": meds},
                    "raw": c["raw"],
                }
            )
        return out

    scored = []
    for c in cases:
        tokens = _tokenize(c["dialogue"])
        score = _jaccard_similarity(q_tokens, tokens)
        if score > 0:
            scored.append((score, c))

    scored.sort(reverse=True, key=lambda x: x[0])
    top = [c for _, c in scored[:max_cases]]

    out = []
    for c in top:
        txt = c["dialogue"]
        cc = _extract_chief_complaint_text(txt)
        sym = _extract_symptom_phrases(txt)
        meds = _extract_medications(txt)
        out.append(
            {
                "chief_complaint": cc,
                "symptoms": sym,
                "medications": meds,
                "drugs": meds,
                "objective": {"medications": meds},
                "raw": c["raw"],
            }
        )

    return out
