"""Verifiable instruction-following adapter for the Rung 4 evolution loop.

The core runner remains untouched so active LiveCodeBench jobs can resume from
their recorded source hashes.  This adapter uses the official IFBench package
for both its OOD constraints and the classic Google IFEval constraints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import rung4
import rung4_benchmark
from ifbench import instructions_registry
from openevolve.config import PromptConfig
from openevolve.prompt.sampler import PromptSampler

IFBENCH_COMMIT = "1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d"
IFEVAL_REVISION = "08a8d6736475776f42ffac23b2c13111a28e5795"

MINIMAL_SKILL = """# Instruction-following procedure
Answer the user's request.
"""
STRONG_SKILL = """# Instruction-following procedure
Read the entire request before answering. Identify every explicit requirement,
including required content, forbidden content, counts, length, language, case,
punctuation, ordering, and output format. Draft the answer, then check it against
each requirement and correct every violation. Return only the requested answer.
"""


def prepare_dataset(output: Path, source: Path, benchmark: str, train_count: int,
                    holdout_count: int, split_seed: int = 0) -> dict:
    """Create a fixed split; holdout_count=-1 uses every remaining example."""
    if benchmark not in {"ifbench", "ifeval"}:
        raise ValueError("benchmark must be ifbench or ifeval")
    rows = [json.loads(line) for line in source.open() if line.strip()]
    if len({str(row["key"]) for row in rows}) != len(rows):
        raise ValueError("Source task IDs must be unique")
    if train_count <= 0 or holdout_count < -1:
        raise ValueError("train_count must be positive; holdout_count must be >= -1")
    if holdout_count == -1:
        holdout_count = len(rows) - train_count
    if holdout_count <= 0:
        raise ValueError("The split must leave at least one held-out example")
    rows.sort(key=lambda row: str(row["key"]))
    random.Random(split_seed).shuffle(rows)
    if train_count + holdout_count > len(rows):
        raise ValueError(f"Requested too many tasks; only {len(rows)} available")
    tasks = []
    for index, row in enumerate(rows[:train_count + holdout_count]):
        tasks.append(dict(id=f'{benchmark}:{row["key"]}',
                          split="train" if index < train_count else "holdout",
                          prompt=row["prompt"],
                          instruction_id_list=row["instruction_id_list"],
                          kwargs=row["kwargs"]))
    manifest = dict(benchmark=benchmark,
                    evaluator="official IFBench loose verifier",
                    evaluator_commit=IFBENCH_COMMIT,
                    source_revision=IFBENCH_COMMIT if benchmark == "ifbench" else IFEVAL_REVISION,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    split_seed=split_seed, train_count=train_count,
                    holdout_count=holdout_count, tasks=tasks)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and json.loads(output.read_text()) != manifest:
        raise FileExistsError("Refusing to replace a different benchmark split")
    output.write_text(json.dumps(manifest))
    return manifest


class InstructionBenchmark:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("benchmark") not in {"ifbench", "ifeval"}:
            raise ValueError("Not an IFBench/IFEval manifest")
        self.tasks = {row["id"]: row for row in self.manifest["tasks"]}
        self.train_ids = [key for key, row in self.tasks.items() if row["split"] == "train"]
        self.holdout_ids = [key for key, row in self.tasks.items() if row["split"] == "holdout"]

    def task(self, task_id):
        return self.tasks[task_id]

    def public_task(self, task_id):
        row = self.task(task_id)
        return {"id": row["id"], "prompt": row["prompt"]}


class NullSandbox:
    def __init__(self, _):
        pass


def _checks(task: dict, response: str, loose: bool) -> list[bool]:
    variants = [response]
    if loose:
        lines = response.split("\n")
        variants = [response, response.replace("*", ""),
                    "\n".join(lines[1:]).strip(), "\n".join(lines[:-1]).strip(),
                    "\n".join(lines[1:-1]).strip()]
        variants += [value.replace("*", "") for value in variants[2:]]
    followed = []
    for instruction_id, raw_kwargs in zip(task["instruction_id_list"], task["kwargs"]):
        instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
        # Released IFBench rows carry a superset of nullable kwargs. The
        # official strict path removes them; doing so for loose evaluation too
        # avoids passing irrelevant keywords to narrower verifier signatures.
        kwargs = {key: value for key, value in raw_kwargs.items() if value is not None}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=task["prompt"])
        followed.append(any(value.strip() and instruction.check_following(value)
                            for value in variants))
    return followed


def score(task: dict, response: str) -> dict:
    loose = _checks(task, response, True)
    strict = _checks(task, response, False)
    total = len(loose)
    passed = sum(loose)
    return dict(reward=float(bool(total) and all(loose)),
                prompt_level_loose=float(bool(total) and all(loose)),
                prompt_level_strict=float(bool(total) and all(strict)),
                constraint_accuracy=passed / total if total else 0.0,
                strict_constraint_accuracy=sum(strict) / total if total else 0.0,
                failed_instruction_ids=[instruction_id for instruction_id, ok in
                    zip(task["instruction_id_list"], loose) if not ok],
                passed_test_fraction_lower_bound=passed / total if total else 0.0,
                passed_tests=passed, total_tests=total, timed_out=False, seconds=0.0)


def execute(model, benchmark, sandbox, task_id: str, files: dict,
            budget, seed: int, phase: str, reserve: int = 0) -> dict:
    """Direct response executor; candidate fitness is loose constraint accuracy."""
    task = benchmark.task(task_id)
    skill = files.get("SKILL.md", "")
    messages = [dict(role="system", content=(
        "Follow the user's instructions precisely. Apply the reusable procedure below "
        "silently, and return only the requested answer.\n\n" + skill)),
        dict(role="user", content=task["prompt"])]
    started = time.monotonic()
    message = model.call(messages, budget, phase, seed, reserve=reserve)
    response = (message or {}).get("content") or ""
    submitted = bool(response.strip())
    measured = score(task, response) if submitted else dict(
        reward=0.0, prompt_level_loose=0.0, prompt_level_strict=0.0,
        constraint_accuracy=0.0, strict_constraint_accuracy=0.0,
        failed_instruction_ids=task["instruction_id_list"],
        passed_test_fraction_lower_bound=0.0, passed_tests=0,
        total_tests=len(task["instruction_id_list"]), timed_out=False,
        seconds=0.0)
    prompt_reward = measured["reward"]
    if phase.startswith("development"):
        measured["reward"] = measured["constraint_accuracy"]
    measured["prompt_level_reward"] = prompt_reward
    return dict(task_id=task_id, submitted=submitted, **measured,
                wall_seconds=time.monotonic() - started, tools=[],
                reflection=dict(failed_instruction_ids=measured["failed_instruction_ids"],
                                constraint_accuracy=measured["constraint_accuracy"],
                                response_excerpt=response[:6000]),
                solution_sha256=hashlib.sha256(response.encode()).hexdigest() if submitted else None)


class InstructionLearner(rung4.Learner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = PromptSampler(PromptConfig(
            system_message=(
                "Improve a reusable Markdown procedure for following ALL explicit constraints "
                "in NEW user requests. Do not answer a particular request or memorize examples. "
                "Return the complete SKILL.md in a markdown code fence."),
            use_template_stochasticity=False))


def configure_runner() -> None:
    rung4.INITIAL_SKILLS = {"minimal": MINIMAL_SKILL, "strong": STRONG_SKILL}
    rung4.Learner = InstructionLearner
    rung4.execute = execute
    rung4_benchmark.LiveCodeBench = InstructionBenchmark
    rung4_benchmark.Sandbox = NullSandbox


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--benchmark", choices=("ifbench", "ifeval"))
    parser.add_argument("--train-count", type=int, default=66)
    parser.add_argument("--holdout-count", type=int, default=16,
                        help="Number of test examples; -1 uses all remaining examples")
    args = parser.parse_args()
    if args.prepare:
        if not args.source or not args.benchmark:
            parser.error("--prepare requires --source and --benchmark")
        manifest = prepare_dataset(args.prepare, args.source, args.benchmark,
                                   args.train_count, args.holdout_count)
        print(json.dumps({key: value for key, value in manifest.items() if key != "tasks"}))
        return
    if not args.config or not args.output:
        parser.error("--config and --output are required for a run")
    configure_runner()
    config = json.loads(args.config.read_text())
    source = Path(__file__).read_bytes()
    config["instruction_adapter"] = dict(sha256=hashlib.sha256(source).hexdigest(),
                                          evaluator_commit=IFBENCH_COMMIT,
                                          development_fitness="loose constraint accuracy",
                                          scored_metric="loose prompt accuracy")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot = args.output / "source" / Path(__file__).name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and snapshot.read_bytes() != source:
        raise ValueError("Refusing resume after instruction adapter source change")
    snapshot.write_bytes(source)
    (rung4.replay if args.replay else rung4.run)(config, args.output)


if __name__ == "__main__":
    main()
