# Day 07 - Reproducibility

## Environment

- Windows PowerShell
- Python virtual environment: `.venv`
- Source corpus: Harris Wang (2023), Introduction to Computer Programming with Python
- Source license: CC BY-NC-SA 4.0 (verify third-party material exceptions)
- Day 06 dependency: local ChromaDB and MiniLM embedding
- Gemini model used in evaluated runs: `gemini-3.5-flash-lite`

## Corpus

Day 07 evaluates 200 by-paragraph chunks, originating from
45 distinct PDF pages (PDF pages 190-235).

The Day 06 ChromaDB collection `lab06_chunks` contains 403 chunks
overall. Dense search applies `strategy=by_paragraph`.

Local input files expected:

- `starter/data/converted/chunks.json`
- `starter/data/converted/questions_full.json`

The benchmark is also included at:
`submission_evidence/benchmark/questions_full.json`

## Retrieval configuration

- Baseline: TF-IDF, top-k=3, context budget 800 characters.
- Dense: MiniLM, 384 dimensions.
- Hybrid: TF-IDF plus dense, Reciprocal Rank Fusion k=60.
- Hybrid candidate depth: 10 per source.
- Selected Hybrid output: top-k=3.
- Selected context: Seed-First neighbor expansion, budget 2400 characters.

## Local setup

1. Create and activate a Python environment.
2. Install project requirements from the existing requirement files.
3. Complete the Day 06 ingestion process for the source PDF.
4. Ensure `DAY6_ROOT` points to the local Day 06 project folder.
5. Verify the Chroma collection `lab06_chunks` exists and has
   embeddings compatible with `MyEmbedder("minilm")`.
6. Copy the benchmark into `starter/data/converted/questions_full.json`
   if the local benchmark file is absent.
7. Set Gemini credentials through local environment configuration.
   Never commit credentials.

## Evaluation

Run unit tests:

    python -m unittest discover -s tests -q

Run Retrieval-only TEST (requires local source chunks and Chroma):

    python eval_retrieval_test.py

Run Agent evaluation (requires Gemini credentials):

    python eval_dev_agent.py
    python eval_test_seedfirst.py
    python eval_test_baseline.py

Important: Existing TEST results are frozen. Do not overwrite
the submitted runs during reproduction. Use a clean copy or
separate output directory if rerunning.

## Submitted evidence

- `submission_evidence/benchmark/`: 24 questions, DEV/TEST.
- `submission_evidence/retrieval/`: retrieval metrics and context review.
- `submission_evidence/agent/`: reviewed metrics and selected traces.
- `submission_evidence/audit/`: absence keyword audits.
- `report/individual/2A202603761_NguyenGiaNam.md`: personal report.

The evidence folder contains reduced agent traces. Complete
execution traces remain in ignored local `starter/results/`.

## Limitations

- The PDF, ChromaDB database, embedding artifacts and full traces
  are not bundled in the submitted repository.
- Exact package versions and external model revisions are not pinned
  in this document.
- Unanswerable-case absence verification is provisional.
- Gemini calls incur API usage and variable network latency.
- The held-out TEST was not used to retune the selected method.
