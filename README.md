# ProcessODPs / Text2ODP

A reproducible Python pipeline to transform scientific text (paper abstracts) into **Ontology Design Patterns (ODPs)**.

The project supports:
1. Downloading abstracts from **arXiv** or **Crossref**.
2. Generating a domain **scenario + competency questions (CQs)** using open-source LLMs.
3. Extracting candidate **concepts and relations**.
4. Generating an **ODP** (classes, object properties, axioms, Turtle).
5. Running an evaluation suite suitable for scientific reporting.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional (for local transformer / embedding workflows):

```bash
pip install -e .[llm]
```

## Open-source LLM options

### 1) Ollama (recommended local baseline)
- Example model: `llama3.1:8b`
- Start Ollama and pull model before running commands.

### 2) Hugging Face Inference API
- Set `HF_TOKEN` in environment.
- Use open models (e.g., Mistral, Llama derivatives with valid access).

## CLI usage

### Step A. Build abstract dataset

```bash
text2odp build-dataset \
  --provider ollama \
  --model llama3.1:8b \
  --query "ontology design pattern process mining" \
  --source arxiv \
  --max-results 30 \
  --output-dir outputs
```

This writes `outputs/abstract_dataset.jsonl`.

### Step B. Generate Text2ODP outputs

```bash
text2odp generate \
  --provider ollama \
  --model llama3.1:8b \
  --dataset-path outputs/abstract_dataset.jsonl \
  --output-dir outputs
```

This writes `outputs/generated_odp_dataset.jsonl`.

### Step C. Evaluate

```bash
text2odp evaluate \
  --generated-path outputs/generated_odp_dataset.jsonl \
  --output-path outputs/evaluation.json
```

## Evaluation design (for publication)

Current metrics:
- **CQ Coverage**: TF-IDF semantic similarity between CQs and generated axioms.
- **Relation Grounding**: fraction of generated object properties that are grounded in extracted relation text.
- **TTL Parse Rate**: percent of generated Turtle snippets that parse with `rdflib`.

These are deterministic and reproducible given fixed generated outputs.

## Suggested scientific protocol

For a publishable study, run:
1. Multiple open models (e.g., Llama 3.1 8B, Mistral 7B, Qwen 7B).
2. Fixed paper set with domain-balanced strata.
3. 3 random seeds/prompts variants per model.
4. Report mean ± std for each metric.
5. Add manual expert evaluation rubric:
   - conceptual correctness,
   - ontology design quality,
   - CQ-answerability.

You can extend `src/text2odp/evaluation.py` with additional metrics such as SHACL constraint checks, ontology reasoner consistency, and inter-annotator agreement integration.

## Project structure

- `src/text2odp/data.py`: abstract acquisition from arXiv/Crossref.
- `src/text2odp/prompts.py`: JSON prompts for each generation stage.
- `src/text2odp/pipeline.py`: end-to-end Text2ODP generation pipeline.
- `src/text2odp/evaluation.py`: reproducible automatic evaluation.
- `src/text2odp/cli.py`: executable interface.

## Reproducibility checklist

- Pin dependencies via lock file in your paper artifact.
- Store exact prompts and model IDs in experiment logs.
- Version the generated datasets (`jsonl`) and evaluation outputs.
- Provide all scripts/commands used for each table/figure.
