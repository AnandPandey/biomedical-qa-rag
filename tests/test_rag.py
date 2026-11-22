# Tests for the RAG pieces: evidence function, --k override, configs and retrieval metrics.
# None of them needs bm25s, the real corpus or an API key (stubs and fakes are used instead).
# Path is used to find the configs folder.
from pathlib import Path

# load_config reads a YAML config into a dict.
from bmqa.config import load_config

# load_predictions reads a predictions file back into Prediction objects.
from bmqa.io import load_predictions

# A fake LLM that returns a fixed reply, records every prompt and never uses the network.
from bmqa.llm.client import FakeLLMClient

# The unchanged Phase 2 runner.
from bmqa.llm.run import execute

# The functions under test.
from bmqa.rag.evidence import make_retrieval_evidence
from bmqa.rag.run import resolve_config
from bmqa.retrieval.evaluate import rank_of, retrieval_metrics

# Shared record types: one question, and one search result.
from bmqa.schema import Example, Hit

# Path to the project's configs/ folder: this file's folder is tests/, and parents[1] is the
# project root one level above it.
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# A fake retriever: always returns documents d1, d2, ... dk in that order, whatever the
# query. This makes test results predictable and avoids needing a real index.
class StubRetriever:
    # Same signature as the real retriever: a query and how many results to return.
    def retrieve(self, query, k):
        # Hit i has doc_id "d<i>", a decreasing score and rank i (1 = best).
        return [Hit(doc_id=f"d{i}", score=10.0 - i, rank=i) for i in range(1, k + 1)]


# A fake document store: a plain dict {"d1": "document text 1", ..., "d5": ...}.
STORE = {f"d{i}": f"document text {i}" for i in range(1, 6)}


# Checks that the evidence function returns the top-k hits paired with their texts.
def test_evidence_function_returns_hits_with_texts():
    # Build an evidence function that returns 2 documents per question.
    evidence = make_retrieval_evidence(StubRetriever(), STORE, k=2)
    # A minimal fake question.
    ex = Example(id="1", question="Q?", contexts=["c"], long_answer="a", final_decision="yes")
    # Call it like the runner would.
    pairs = evidence(ex)
    # We get exactly d1 and d2, each with the right text from the store.
    assert [(h.doc_id, text) for h, text in pairs] == [
        ("d1", "document text 1"),
        ("d2", "document text 2"),
    ]


# End-to-end test of a RAG run with the fake LLM: retrieved texts reach the prompt and the
# output is saved under the renamed system. processed_dir is a fixture (tests/conftest.py)
# that writes a tiny train/dev/test split; tmp_path is a temporary folder.
def test_rag_run_with_fake_llm(tmp_path, processed_dir):
    # Load the real strict config and override k to 3 (so the system becomes rag_strict_k3).
    cfg = resolve_config(load_config(CONFIGS / "rag_strict_bm25.yaml"), 3)
    # The fake LLM; we keep a reference to inspect the prompts it received.
    client = FakeLLMClient()
    # Run the Phase 2 runner; "_" ignores the second returned path (the usage log).
    pred_path, _ = execute(
        # The config with k = 3.
        cfg,
        # Retrieval evidence from the stub retriever and fake store, cfg["k"] = 3 docs.
        make_retrieval_evidence(StubRetriever(), STORE, cfg["k"]),
        # Use the tiny dev split.
        split="dev",
        # First run.
        run=1,
        # Read questions from the fixture's folder instead of data/processed.
        processed_dir=processed_dir,
        # Write results into the temporary folder, not the real results/ folder.
        results_dir=tmp_path,
        # Use the fake LLM instead of a real API client.
        client=client,
    )
    # Read the saved predictions back.
    preds = load_predictions(pred_path)
    # The output folder is named after the renamed system, so ablations never overwrite.
    assert pred_path.parent.name == "rag_strict_k3"
    # The first prediction records exactly the 3 retrieved documents as its evidence.
    assert [h.doc_id for h in preds[0].evidence] == ["d1", "d2", "d3"]
    # The third document's text was put into the prompt, numbered [3].
    assert "[3] document text 3" in client.prompts[0]


# Checks resolve_config: no override returns the same object; an override renames the
# system and sets k. A different k means a different config, so the config hash changes too.
def test_k_override_renames_system_and_changes_hash():
    # A minimal config.
    cfg = {"system": "rag_gold", "k": 5}
    # Without --k the very same dict object is returned ("is" checks identity, not equality).
    assert resolve_config(cfg, None) is cfg
    # With --k 10 we get a new dict with k = 10 and system rag_gold_k10.
    assert resolve_config(cfg, 10) == {"system": "rag_gold_k10", "k": 10}


# Checks that both RAG configs use exactly the same LLM settings as closed_book.yaml.
# This makes the comparison fair: the ONLY difference between closed_book and RAG is the
# retrieved evidence, so any change in accuracy is caused by retrieval, not by a different
# model, prompt or setting.
def test_rag_configs_share_llm_settings_with_closed_book():
    # Load the Phase 2 closed-book config (no evidence).
    closed = load_config(CONFIGS / "closed_book.yaml")
    # Check both RAG configs.
    for name in ("rag_strict_bm25.yaml", "rag_gold_bm25.yaml"):
        # Load this RAG config.
        rag = load_config(CONFIGS / name)
        # Compare the model settings, the prompt version and the prices.
        for key in ("llm", "prompt_version", "price_per_million_tokens"):
            # Must be identical; the message after the comma is shown if the check fails.
            assert rag[key] == closed[key], f"{name}: {key} differs from closed_book.yaml"
    # Load both RAG configs again to compare k.
    strict = load_config(CONFIGS / "rag_strict_bm25.yaml")
    gold = load_config(CONFIGS / "rag_gold_bm25.yaml")
    # Strict and gold must retrieve the same number of documents, so they differ only in
    # whether the question's own article is in the corpus.
    assert strict["k"] == gold["k"]


# Checks rank_of and the Recall@k / MRR calculation on hand-computed numbers.
def test_retrieval_metrics():
    # A ranked list of three doc_ids, best first.
    ranked = ["a", "b", "c"]
    # "b" is second (rank 2); "z" is not in the list (None).
    assert rank_of("b", ranked) == 2 and rank_of("z", ranked) is None
    # Four questions whose correct document was at rank 1, rank 3, not found, rank 2.
    m = retrieval_metrics([1, 3, None, 2], ks=(1, 3))
    # Recall@1: only 1 of 4 is at rank 1 -> 0.25. Recall@3: 3 of 4 are in the top 3 -> 0.75.
    assert m["recall@1"] == 0.25 and m["recall@3"] == 0.75
    # MRR@3: average of 1/1, 1/3, 0 (not found) and 1/2 over 4 questions.
    assert m["mrr@3"] == (1 + 1 / 3 + 1 / 2) / 4
