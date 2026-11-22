"""tools/compare_runs.py: two runs are compared question by question."""

import json

import pytest

from tools import compare_runs


def row(number, lost, passages=1, rank=5):
    return {
        "id": number,
        "domain": "DONES",
        "question_type": "factual",
        "passages": [{}] * passages,
        "lost": lost,
        "rank": rank,
    }


def write_run(folder, name, rows):
    run = folder / name
    run.mkdir(parents=True)
    lines = "".join(json.dumps(each) + "\n" for each in rows)
    (run / "questions.jsonl").write_text(lines, encoding="utf-8")
    return run


A = [row(1, None), row(2, None), row(3, "ranking", rank=80), row(4, "parsing", rank=None)]
B = [row(1, None), row(2, "ranking", rank=60), row(3, None), row(4, None)]
A.append(row(5, None, passages=0))  # no passage in the corpus: not a question to compare on
B.append(row(5, None, passages=0))


def by_id(rows):
    return {each["id"]: each for each in rows}


def test_questions_are_split_by_which_run_covers_them():
    result = compare_runs.compare(by_id(A), by_id(B))
    assert result["questions"] == 4
    assert (result["both"], result["only_a"], result["only_b"]) == ([1], [2], [3, 4])
    assert result["neither"] == []


def test_a_question_without_passages_counts_for_neither_run():
    assert not compare_runs.covered(row(5, None, passages=0))
    assert compare_runs.covered(row(1, None))


def test_the_sign_test_is_exact():
    assert compare_runs.sign_test(0, 0) == 1.0
    assert compare_runs.sign_test(1, 2) == 1.0
    # Five questions all one way: twice one in thirty-two.
    assert compare_runs.sign_test(0, 5) == pytest.approx(2 / 32)
    assert compare_runs.sign_test(5, 0) == pytest.approx(2 / 32)
    # 1 against 9: twice (1 + 10) in 1024.
    assert compare_runs.sign_test(1, 9) == pytest.approx(22 / 1024)
    assert compare_runs.sign_test(4, 4) == 1.0


def test_runs_over_different_questions_are_not_compared():
    with pytest.raises(ValueError, match="not over the same questions"):
        compare_runs.compare(by_id(A), by_id(B[:3]))


def test_a_run_is_named_by_the_start_of_its_id(tmp_path):
    write_run(tmp_path, "20261009T040403Z-257534f-935a", A)
    write_run(tmp_path, "20261009T050000Z-aa028a1-1b2c", B)
    name, rows = compare_runs.load("20261009T04", tmp_path)
    assert name == "20261009T040403Z-257534f-935a" and sorted(rows) == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="names 2 runs"):
        compare_runs.load("20261009", tmp_path)
    with pytest.raises(ValueError, match="names 0 runs"):
        compare_runs.load("2027", tmp_path)


def test_the_command_lists_the_questions_the_runs_differ_on(tmp_path, capsys):
    write_run(tmp_path, "run-a", A)
    write_run(tmp_path, "run-b", B)
    assert compare_runs.main(["run-a", "run-b", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "4 questions with a passage: A covers 2, B covers 3" in out
    assert "both 1, only A 1, only B 2, neither 0" in out
    assert "only A: 2 (DONES, factual); the other: lost in ranking (rank 60)" in out
    assert "only B: 4 (DONES, factual); the other: lost in parsing (rank None)" in out

    with pytest.raises(SystemExit):
        compare_runs.main(["run-a", "no-such-run", "--runs", str(tmp_path)])


def mark(number, verdict, domain="DONES"):
    return {"id": number, "domain": domain, "question_type": "factual", "verdict": verdict}


def write_marks(folder, name, rows, judge="second-model"):
    run = folder / name
    run.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(each) + "\n" for each in rows)
    (run / f"judged-{judge}.jsonl").write_text(lines, encoding="utf-8")
    return run


MARKED_A = [mark(1, "correct"), mark(2, "correct")]
MARKED_A += [mark(3, "incorrect", "AI Papers"), mark(4, "unclear", "AI Papers")]
MARKED_B = [mark(1, "correct"), mark(2, "incorrect")]
MARKED_B += [mark(3, "correct", "AI Papers"), mark(4, "correct", "AI Papers")]


def test_answers_are_split_by_which_run_is_right():
    result = compare_runs.compare(by_id(MARKED_A), by_id(MARKED_B), marks=True)
    assert result["questions"] == 4
    assert (result["both"], result["only_a"], result["only_b"]) == ([1], [2], [3, 4])
    assert result["neither"] == []
    # An answer the judge could not mark is not a right one.
    assert not compare_runs.right(mark(4, "unclear"))


def test_the_marks_read_are_those_of_one_judge(tmp_path):
    write_marks(tmp_path, "run-a", MARKED_A)
    name, rows = compare_runs.load_marks("run-a", tmp_path)
    assert name == "run-a" and sorted(rows) == [1, 2, 3, 4]
    write_marks(tmp_path, "run-a", MARKED_B, judge="another")
    with pytest.raises(ValueError, match="has 2 files of marks"):
        compare_runs.load_marks("run-a", tmp_path)
    assert compare_runs.load_marks("run-a", tmp_path, "another")[1][2]["verdict"] == "incorrect"
    write_run(tmp_path, "run-c", A)
    with pytest.raises(ValueError, match="has 0 files of marks"):
        compare_runs.load_marks("run-c", tmp_path)


def test_the_command_compares_two_runs_of_answers_by_their_marks(tmp_path, capsys):
    write_marks(tmp_path, "run-a", MARKED_A)
    write_marks(tmp_path, "run-b", MARKED_B)
    assert compare_runs.main(["run-a", "run-b", "--marks", "--runs", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "4 questions: A is right on 2, B on 3" in out
    assert "both 1, only A 1, only B 2, neither 0" in out
    assert "[compare] AI Papers: only A 0, only B 2\n" in out
    assert "[compare] DONES: only A 1, only B 0\n" in out
    assert "only A: 2 (DONES, factual)\n" in out
    assert "only B: 4 (AI Papers, factual)\n" in out
