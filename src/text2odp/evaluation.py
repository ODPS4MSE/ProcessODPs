from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
from rdflib import Graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schemas import GeneratedSample


@dataclass
class SampleEvaluation:
    paper_id: str
    cq_coverage: float
    relation_grounding: float
    ttl_parse_success: bool


class ODPEvaluator:
    """Evaluation suite with reproducible, publishable-friendly metrics."""

    def evaluate(self, samples: Iterable[GeneratedSample]) -> List[SampleEvaluation]:
        results: List[SampleEvaluation] = []
        for sample in samples:
            cqs = sample.scenario_and_cqs.competency_questions
            axioms = sample.odp.axioms
            cq_coverage = _semantic_coverage(cqs, axioms)
            relation_grounding = _relation_grounding(sample.extraction.relations, sample.odp.object_properties)
            ttl_ok = _parse_turtle(sample.odp.turtle)
            results.append(
                SampleEvaluation(
                    paper_id=sample.paper.paper_id,
                    cq_coverage=float(cq_coverage),
                    relation_grounding=float(relation_grounding),
                    ttl_parse_success=ttl_ok,
                )
            )
        return results

    @staticmethod
    def aggregate(results: Iterable[SampleEvaluation]) -> dict:
        res = list(results)
        if not res:
            return {"n": 0}
        return {
            "n": len(res),
            "mean_cq_coverage": float(np.mean([r.cq_coverage for r in res])),
            "mean_relation_grounding": float(np.mean([r.relation_grounding for r in res])),
            "ttl_parse_rate": float(np.mean([1.0 if r.ttl_parse_success else 0.0 for r in res])),
        }


def _semantic_coverage(cqs: List[str], axioms: List[str]) -> float:
    if not cqs or not axioms:
        return 0.0
    corpus = cqs + axioms
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    cq_vectors = matrix[: len(cqs)]
    ax_vectors = matrix[len(cqs) :]
    sims = cosine_similarity(cq_vectors, ax_vectors)
    return float(np.mean(np.max(sims, axis=1)))


def _relation_grounding(relations: List[str], object_properties: List[str]) -> float:
    if not relations or not object_properties:
        return 0.0
    rel_text = " ".join(relations).lower()
    matches = sum(1 for prop in object_properties if prop.lower() in rel_text)
    return matches / max(len(object_properties), 1)


def _parse_turtle(turtle: str) -> bool:
    g = Graph()
    try:
        g.parse(data=turtle, format="turtle")
        return True
    except Exception:
        return False
