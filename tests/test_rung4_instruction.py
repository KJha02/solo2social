import json

import rung4_instruction as instruction


def test_instruction_score_exposes_dense_fitness_and_prompt_accuracy():
    task = dict(prompt="Answer without commas", instruction_id_list=[
        "punctuation:no_comma"], kwargs=[{}])
    correct = instruction.score(task, "OK")
    assert correct["reward"] == correct["constraint_accuracy"] == 1
    wrong = instruction.score(task, "No, thanks")
    assert wrong["reward"] == wrong["constraint_accuracy"] == 0


def test_instruction_manifest_split_is_fixed_and_disjoint(tmp_path):
    source = tmp_path / "source.jsonl"
    rows = [dict(key=i, prompt=f"prompt {i}",
                 instruction_id_list=["punctuation:no_comma"], kwargs=[{}])
            for i in range(10)]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "manifest.json"
    first = instruction.prepare_dataset(output, source, "ifeval", 6, 2, 7)
    second = instruction.prepare_dataset(output, source, "ifeval", 6, 2, 7)
    assert first == second
    benchmark = instruction.InstructionBenchmark(output)
    assert len(benchmark.train_ids) == 6 and len(benchmark.holdout_ids) == 2
    assert set(benchmark.train_ids).isdisjoint(benchmark.holdout_ids)
    remainder = instruction.prepare_dataset(tmp_path / "rest.json", source,
                                            "ifeval", 6, -1, 7)
    assert remainder["train_count"] == 6
    assert remainder["holdout_count"] == 4
    assert len(remainder["tasks"]) == len(rows)
    assert remainder["tasks"][:6] == first["tasks"][:6]
    assert {task["id"] for task in remainder["tasks"]} == {
        f"ifeval:{row['key']}" for row in rows}
