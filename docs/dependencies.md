# Dependencies and adaptations

## OpenEvolve

We use upstream OpenEvolve at commit
`411fb59c886c18704caaffb611e17cf9e7d824d2` (Apache-2.0).
The installed Python package used in the research workspace was compared with
that upstream tree: source files and prompt templates match. No vendored fork
or external patch is needed.

Our adapter is in `code/rung4.py`, `code/rung4_instruction.py`, and
`code/rung4_population.py`. It evolves Markdown skill files rather than model
weights. OpenEvolve supplies parent sampling, scored candidate archives,
islands, revision prompts, and rewrite parsing. Our code supplies the task
executor, social acquisition and deployment decisions, API transport, budget
accounting, and resumable evaluation. The population runner restores archive
insertion order, set mutation history, and random state to preserve parent
selection across interruption. Run with `PYTHONHASHSEED=0`.

The full solo OpenEvolve condition uses the archive to select parents and deploy
the highest-scoring skill. LLM-controlled conditions choose their learning
action and deployed file themselves. Copying acquires a peer's previously
deployed file; it does not force deployment. Archive migrations occur only
within an agent, never across agents.

## IFBench

The evaluator and source data come from upstream IFBench commit
`1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d` (Apache-2.0).
The installed evaluator source also matches upstream. Dataset preparation sorts
source keys, shuffles with seed 0, and takes 100 training and 200 test examples.
The original source JSONL SHA-256 is
`d2ada7da94a38cfe406351614c4e686846ed2da6d1b339db95fa5ead19554a4a`.
Preparation downloads `data/IFBench_test.jsonl` from that exact commit and
verifies this checksum. The copy bundled inside the upstream Python package
has a different checksum and is not the source used for this split.
Both individual-constraint and whole-answer results use IFBench's loose verifier.

IFBench's checks need NLTK resources. `runners/prepare_data.py` prepares them
under ignored `data/nltk_data/`; allow network access during setup. Do not
reinterpret evaluator failures as incorrect model answers.

## Other dependencies

`uv.lock` records the resolved environment. The optional vLLM stack preserves
the model-specific decoding settings in `code/main.py`; installing the API-only
extra does not require that GPU stack. Package and dataset licenses remain with
their respective projects; no third-party package source or model weights are
redistributed here.

The coding-task adapter used in earlier supporting diagnostics is retained as
`code/rung4_benchmark.py`. It requires a separately prepared Apptainer sandbox
and the pinned official LiveCodeBench evaluator, not the IFBench null sandbox.
See `docs/reproduction.md` before using it. The paper's principal population
experiment is static IFBench, not the earlier streaming design.
