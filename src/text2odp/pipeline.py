from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from .data import AbstractCollector
from .llm_client import LLMClient
from .prompts import EXTRACTION_PROMPT, ODP_PROMPT, SCENARIO_AND_CQ_PROMPT
from .schemas import ConceptRelationExtraction, GeneratedSample, ODPResult, PaperRecord, ScenarioAndCQs


@dataclass
class PipelineConfig:
    output_dir: Path = Path("outputs")


class Text2ODPPipeline:
    def __init__(self, llm_client: LLMClient, config: PipelineConfig | None = None) -> None:
        self.llm = llm_client
        self.collector = AbstractCollector()
        self.config = config or PipelineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(self, query: str, max_results: int = 30, source: str = "arxiv") -> List[PaperRecord]:
        if source.lower() == "arxiv":
            records = self.collector.fetch_from_arxiv(query=query, max_results=max_results)
        elif source.lower() == "crossref":
            records = self.collector.fetch_from_crossref(query=query, rows=max_results)
        else:
            raise ValueError(f"Unsupported source: {source}")

        out = self.config.output_dir / "abstract_dataset.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(record.model_dump_json() + "\n")
        return records

    def run_generation(self, records: Iterable[PaperRecord]) -> List[GeneratedSample]:
        generated: List[GeneratedSample] = []
        for record in tqdm(list(records), desc="Generating ODPs"):
            scenario_json = self.llm.generate_json(SCENARIO_AND_CQ_PROMPT.format(abstract=record.abstract))
            scenario = ScenarioAndCQs(**scenario_json)

            extraction_json = self.llm.generate_json(
                EXTRACTION_PROMPT.format(
                    scenario=scenario.scenario,
                    cqs="\n".join(f"- {q}" for q in scenario.competency_questions),
                )
            )
            extraction = ConceptRelationExtraction(**extraction_json)

            odp_json = self.llm.generate_json(
                ODP_PROMPT.format(
                    concepts="\n".join(f"- {c}" for c in extraction.concepts),
                    relations="\n".join(f"- {r}" for r in extraction.relations),
                )
            )
            odp = ODPResult(**odp_json)

            generated.append(
                GeneratedSample(
                    paper=record,
                    scenario_and_cqs=scenario,
                    extraction=extraction,
                    odp=odp,
                )
            )

        out = self.config.output_dir / "generated_odp_dataset.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for item in generated:
                f.write(item.model_dump_json() + "\n")
        return generated

    @staticmethod
    def load_papers(path: Path) -> List[PaperRecord]:
        records: List[PaperRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(PaperRecord(**json.loads(line)))
        return records
