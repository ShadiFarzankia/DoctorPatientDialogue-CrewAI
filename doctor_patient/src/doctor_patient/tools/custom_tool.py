# src/doctor_patient/tools/custom_tool.py
from __future__ import annotations

from typing import List, Type

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from . import retrieval


# Make sure the JSON is loaded once at startup (idempotent).
retrieval.build_stores()


# ------------- Tool 1: get_candidate_symptoms ------------- #

class GetCandidateSymptomsInput(BaseModel):
    chief_complaint: str = Field(
        ...,
        description=(
            "The patient's free-text chief complaint, e.g. "
            "'I have had a cough and fever for 3 days'."
        ),
    )


class GetCandidateSymptoms(BaseTool):
    name: str = "get_candidate_symptoms"
    description: str = (
        "Retrieve candidate acute symptoms from ACI-Bench train_full.json "
        "that match the patient's chief complaint."
    )
    args_schema: Type[BaseModel] = GetCandidateSymptomsInput

    def _run(self, chief_complaint: str) -> List[dict]:
        return retrieval.get_candidate_symptoms_for_chief_complaint(chief_complaint)


# ------------- Tool 2: get_candidate_drugs ------------- #

class GetCandidateDrugsInput(BaseModel):
    selected_symptoms: List[str] = Field(
        ...,
        description=(
            "List of confirmed symptom names, e.g. ['cough', 'shortness of breath']."
        ),
    )


class GetCandidateDrugs(BaseTool):
    name: str = "get_candidate_drugs"
    description: str = (
        "Retrieve commonly reported drugs/medications from similar cases in "
        "train_full.json for the given symptoms."
    )
    args_schema: Type[BaseModel] = GetCandidateDrugsInput

    def _run(self, selected_symptoms: List[str]) -> List[dict]:
        return retrieval.get_candidate_drugs_for_symptoms(selected_symptoms)


# ------------- Tool 3: get_similar_cases_for_summary ------------- #

class GetSimilarCasesForSummaryInput(BaseModel):
    selected_symptoms: List[str] = Field(
        ...,
        description=(
            "List of confirmed symptom names, e.g. ['cough', 'fever'] "
            "used to retrieve similar cases."
        ),
    )
    selected_drugs: List[str] = Field(
        ...,
        description=(
            "List of confirmed drug names, e.g. ['ibuprofen'], used to refine "
            "the retrieval of similar cases."
        ),
    )


class GetSimilarCasesForSummary(BaseTool):
    name: str = "get_similar_cases_for_summary"
    description: str = (
        "Retrieve similar patient cases from train_full.json, including their "
        "Objective and Plan sections, for use in the final clinical-style summary."
    )
    args_schema: Type[BaseModel] = GetSimilarCasesForSummaryInput

    def _run(
        self,
        selected_symptoms: List[str],
        selected_drugs: List[str],
    ) -> List[dict]:
        return retrieval.get_similar_cases_for_summary(
            selected_symptoms=selected_symptoms,
            selected_drugs=selected_drugs,
        )


__all__ = [
    "GetCandidateSymptoms",
    "GetCandidateDrugs",
    "GetSimilarCasesForSummary",
]
