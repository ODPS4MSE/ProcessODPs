from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    abstract: str
    source: str = Field(description="Data source, e.g. arXiv/Crossref")
    year: int | None = None
    doi: str | None = None


class ScenarioAndCQs(BaseModel):
    scenario: str
    competency_questions: List[str]


class ConceptRelationExtraction(BaseModel):
    concepts: List[str]
    relations: List[str] = Field(description="Triples in natural-language form")


class ODPResult(BaseModel):
    pattern_name: str
    intent: str
    classes: List[str]
    object_properties: List[str]
    axioms: List[str]
    turtle: str


class GeneratedSample(BaseModel):
    paper: PaperRecord
    scenario_and_cqs: ScenarioAndCQs
    extraction: ConceptRelationExtraction
    odp: ODPResult
