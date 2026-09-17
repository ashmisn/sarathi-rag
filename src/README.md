# Source module ownership

The current app is intentionally import-compatible with the original prototype. Its responsibilities are separated by module even though they are not yet separate Python packages:

- `data_processor.py`: ingestion, SQLite table creation, combined-text construction, and Chroma indexing.
- `tools.py`: structured SQL, keyword extraction/search, and semantic vector retrieval.
- `agent.py`: router, planner execution, extraction chain, and answer-generation chain.
- `prompts.py`: structured-output schemas and grounding prompts.
- `config.py`: paths, model names, and environment overrides.

The `evaluation/` package is kept outside this runtime path so benchmark runs do not initialize Ollama or download embeddings. A future package refactor should preserve these boundaries and the benchmark JSON contract.
