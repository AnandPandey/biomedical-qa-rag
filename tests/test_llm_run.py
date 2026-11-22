# Tests for the LLM runner (bmqa/llm/run.py). They use FakeLLMClient only, so they need no
# API key, make no network calls and cost nothing. Run them with `pytest`.
# json is used to round-trip predictions and to read the usage log file.
import json

# asdict turns a dataclass object (like Prediction) into a plain dict.
from dataclasses import asdict

# The marker that flags an unparseable answer ("PARSE_FAILURE:").
from bmqa.eval.metrics import PARSE_FAILURE_PREFIX

# Reads a predictions JSONL file back into Prediction objects.
from bmqa.io import load_predictions

# The fake client and the factory that chooses which client to create.
from bmqa.llm.client import FakeLLMClient, make_client

# The runner functions under test, plus the two evidence functions.
from bmqa.llm.run import execute, gold_evidence, no_evidence, run_system

# Shared data types used to build test data and check results.
from bmqa.schema import Example, Prediction


# Helper: three made-up examples with ids "0", "1", "2". Each long_answer holds a unique
# "SECRET" marker so a test can prove the conclusion is never shown to the model.
def _examples():
    return [
        Example(
            id=str(i),
            question=f"Question {i}?",
            contexts=[f"context {i}a", f"context {i}b"],  # two abstract paragraphs
            long_answer=f"SECRET-CONCLUSION-{i}",  # must never appear in a prompt
            final_decision="yes",
        )
        # Build one Example for each i = 0, 1, 2.
        for i in range(3)
    ]


# Helper: run the system on the three examples with fixed settings. Inputs: a client and an
# evidence function (closed-book by default). Output: (predictions, usage) from run_system.
def _run(client, evidence_fn=no_evidence):
    return run_system(
        _examples(),
        client,
        evidence_fn,
        system="closed_book",
        prompt_version="v1",
        config_hash="abc",  # dummy hash; any text is fine for tests
        run=1,
        max_tokens=100,
    )


# With the default fake reply ("yes"), every prediction is valid, in input order and
# survives saving/loading as JSON.
def test_predictions_are_valid_and_in_order():
    preds, usage = _run(FakeLLMClient())
    # Predictions come back in the same order as the examples.
    assert [p.id for p in preds] == ["0", "1", "2"]
    for p in preds:
        # Convert to dict -> JSON text -> dict -> Prediction: the result must equal the
        # original, proving the row can be written to and read from a file without loss.
        assert Prediction.from_dict(json.loads(json.dumps(asdict(p)))) == p
        # The fake's "yes" is used, the fake model name is recorded, and closed-book rows
        # have no evidence documents.
        assert p.prediction == "yes" and p.model == "fake-llm" and p.evidence == []
    # Three answers were counted and none failed to parse.
    assert usage["n"] == 3 and usage["parse_failures"] == 0


# Leakage check: even in the oracle system, the gold conclusion (long_answer) must never be
# part of the prompt, otherwise the model could simply copy the answer.
def test_long_answer_never_reaches_the_prompt():
    client = FakeLLMClient()
    # Run with the oracle evidence function; the fake client records every prompt.
    _run(client, gold_evidence)
    # Some prompts were sent, and none of them contains the SECRET marker.
    assert client.prompts and all("SECRET-CONCLUSION" not in p for p in client.prompts)


# The oracle's single evidence document is the example's own abstract, at rank 1.
def test_oracle_evidence_is_the_example_itself():
    preds, _ = _run(FakeLLMClient(), gold_evidence)
    # Example "0" has exactly one evidence Hit: document "0" at rank 1.
    assert [(h.doc_id, h.rank) for h in preds[0].evidence] == [("0", 1)]


# A reply that is not JSON is kept as a parse failure (not dropped), so it counts as wrong.
def test_unparseable_reply_becomes_parse_failure():
    preds, usage = _run(FakeLLMClient(reply="I think yes"))
    # All three replies failed to parse.
    assert usage["parse_failures"] == 3
    # Each raw_output starts with the "PARSE_FAILURE:" marker the scorer looks for.
    assert all(p.raw_output.startswith(PARSE_FAILURE_PREFIX) for p in preds)
    # The placeholder label "maybe" is stored and there is no rationale.
    assert all(p.prediction == "maybe" and p.rationale is None for p in preds)


# --dry-run must give the free fake client, even when the config names a real provider.
def test_dry_run_uses_fake_client():
    assert isinstance(make_client({"provider": "anthropic"}, run=1, dry_run=True), FakeLLMClient)


# End-to-end: execute() writes a predictions file and a usage log. `tmp_path` is an empty
# temporary folder; `processed_dir` (from tests/conftest.py) is a tiny synthetic dataset
# whose dev split holds one example, id "90000005".
def test_execute_writes_predictions_and_log(tmp_path, processed_dir):
    # A minimal config, like configs/closed_book.yaml.
    cfg = {
        "system": "closed_book",
        "evidence": "none",
        "prompt_version": "v1",
        "llm": {"provider": "anthropic", "model": "m", "effort": "low", "max_tokens": 50},
        # Made-up prices, so the cost estimate is above zero.
        "price_per_million_tokens": {"input": 1.0, "output": 2.0},
    }
    # Run on the dev split with the fake client, writing into the temporary folder.
    pred_path, log_path = execute(
        cfg,
        no_evidence,
        split="dev",
        run=1,
        processed_dir=processed_dir,  # read the synthetic data
        results_dir=tmp_path,  # write outputs into the temporary folder
        client=FakeLLMClient(),  # never call the real API in tests
    )
    # The predictions file holds exactly the one dev example.
    assert [p.id for p in load_predictions(pred_path)] == ["90000005"]
    # Read the usage log JSON file back.
    log = json.loads(log_path.read_text(encoding="utf-8"))
    # One answer was logged, and since this is not a dry run, a cost above 0 was estimated.
    assert log["n"] == 1 and log["estimated_cost_usd"] > 0
