from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .evaluation import ODPEvaluator
from .llm_client import HuggingFaceTextGenClient, OllamaClient
from .pipeline import PipelineConfig, Text2ODPPipeline
from .schemas import GeneratedSample


def build_client(args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaClient(model=args.model, base_url=args.base_url)
    if args.provider == "huggingface":
        token = os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN is required for Hugging Face provider")
        return HuggingFaceTextGenClient(model_id=args.model, hf_token=token)
    raise ValueError(f"Unsupported provider: {args.provider}")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    pipeline = Text2ODPPipeline(llm_client=build_client(args), config=PipelineConfig(output_dir=Path(args.output_dir)))
    records = pipeline.build_dataset(query=args.query, max_results=args.max_results, source=args.source)
    print(f"Saved {len(records)} abstracts to {args.output_dir}/abstract_dataset.jsonl")


def cmd_generate(args: argparse.Namespace) -> None:
    pipeline = Text2ODPPipeline(llm_client=build_client(args), config=PipelineConfig(output_dir=Path(args.output_dir)))
    records = pipeline.load_papers(Path(args.dataset_path))
    generated = pipeline.run_generation(records)
    print(f"Generated {len(generated)} ODPs to {args.output_dir}/generated_odp_dataset.jsonl")


def cmd_evaluate(args: argparse.Namespace) -> None:
    path = Path(args.generated_path)
    samples = [GeneratedSample(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    evaluator = ODPEvaluator()
    results = evaluator.evaluate(samples)
    aggregate = evaluator.aggregate(results)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"aggregate": aggregate, "per_sample": [r.__dict__ for r in results]}, indent=2), encoding="utf-8")
    print(f"Saved evaluation to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="text2odp", description="Text2ODP scientific pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--provider", choices=["ollama", "huggingface"], default="ollama")
    common.add_argument("--model", default="llama3.1:8b")
    common.add_argument("--base-url", default="http://localhost:11434")
    common.add_argument("--output-dir", default="outputs")

    p_dataset = sub.add_parser("build-dataset", parents=[common])
    p_dataset.add_argument("--query", required=True)
    p_dataset.add_argument("--source", choices=["arxiv", "crossref"], default="arxiv")
    p_dataset.add_argument("--max-results", type=int, default=20)
    p_dataset.set_defaults(func=cmd_build_dataset)

    p_generate = sub.add_parser("generate", parents=[common])
    p_generate.add_argument("--dataset-path", required=True)
    p_generate.set_defaults(func=cmd_generate)

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--generated-path", required=True)
    p_eval.add_argument("--output-path", default="outputs/evaluation.json")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
