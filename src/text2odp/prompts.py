SCENARIO_AND_CQ_PROMPT = """
You are an ontology engineering assistant.
Given a paper abstract, generate:
1) a concise domain scenario,
2) 5-10 competency questions (CQs) that an ontology should answer.

Return strict JSON with fields:
{
  "scenario": "...",
  "competency_questions": ["...", "..."]
}

ABSTRACT:
{abstract}
""".strip()


EXTRACTION_PROMPT = """
You are an ontology modeling assistant.
Given a scenario and competency questions, extract:
1) key domain concepts/classes,
2) candidate relations among concepts in short textual triple-like forms.

Return strict JSON with fields:
{
  "concepts": ["...", "..."],
  "relations": ["ConceptA --relation--> ConceptB", "..."]
}

SCENARIO:
{scenario}

COMPETENCY QUESTIONS:
{cqs}
""".strip()


ODP_PROMPT = """
You are an ontology design pattern expert.
Generate an Ontology Design Pattern (ODP) from provided concepts and relations.

Return strict JSON with fields:
{
  "pattern_name": "...",
  "intent": "...",
  "classes": ["...", "..."],
  "object_properties": ["...", "..."],
  "axioms": ["Manchester or readable logical axioms"],
  "turtle": "@prefix ..."
}

CONCEPTS:
{concepts}

RELATIONS:
{relations}
""".strip()
