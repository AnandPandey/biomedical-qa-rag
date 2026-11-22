# Tests for the claim checker (bmqa.eval.claims) and annotation helpers (bmqa.eval.annotate).
# They never load the real NLI model: fake probability dicts and a fake scorer stand in for it,
# so the tests are fast and check only OUR logic.

# Helpers from annotate.py under test.
from bmqa.eval.annotate import label_report, stratified_sample

# Helpers from claims.py under test.
from bmqa.eval.claims import check_claims, decide, fabricated_citations, split_claims, summarize

# Hit = one retrieved document; Prediction = one system answer (shared project data types).
from bmqa.schema import Hit, Prediction

# Fake NLI output where "entailment" is the most likely label (premise supports the claim).
E = {"entailment": 0.8, "neutral": 0.1, "contradiction": 0.1}
# Fake NLI output where "neutral" wins (premise says nothing about the claim).
N = {"entailment": 0.1, "neutral": 0.8, "contradiction": 0.1}
# Fake NLI output where "contradiction" wins (premise conflicts with the claim).
C = {"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8}


# Helper that builds a Prediction with n_docs fake retrieved documents d1..dn.
# Inputs: id, raw LLM reply, rationale text, number of documents, system name.
def _pred(pid, raw, rationale, n_docs, system="rag_strict"):
    # Fake hits "d1", "d2", ... with ranks 1..n_docs (n_docs = 0 gives an empty list).
    hits = [Hit(doc_id=f"d{i}", score=1.0, rank=i) for i in range(1, n_docs + 1)]
    # Build and return the Prediction; fields not under test get dummy values.
    return Prediction(
        id=pid,  # question id
        system=system,  # system name
        prediction="yes",  # predicted answer (not used by these tests)
        raw_output=raw,  # the raw reply, parsed again to find citations
        model="m",  # dummy model name
        config_hash="c",  # dummy config fingerprint
        rationale=rationale,  # the explanation that is split into claims
        evidence=hits,  # documents the model was shown
    )


# Sentence splitting must keep sentences of 4+ words and drop short ones.
def test_split_claims():
    # Three sentences: 5 words, 1 word ("Yes."), 7 words.
    text = "Aspirin lowered the risk clearly. Yes. The effect was seen in older adults!"
    # "Yes." has fewer than 4 words, so only the other two remain, punctuation kept.
    assert split_claims(text) == [
        "Aspirin lowered the risk clearly.",
        "The effect was seen in older adults!",
    ]
    # No rationale (None) or an empty one gives no claims.
    assert split_claims(None) == [] and split_claims("") == []


# The decision rule: entailment anywhere wins, then contradiction, else not_supported.
def test_decide_rule():
    # Verdicts neutral + entailment -> one supporting premise is enough.
    assert decide([N, E]) == "supported"
    # Verdicts neutral + contradiction -> no support, one contradiction.
    assert decide([N, C]) == "contradicted"
    # Verdicts entailment + contradiction -> support wins over contradiction.
    assert decide([E, C]) == "supported"
    # Both neutral -> not supported.
    assert decide([N, N]) == "not_supported"


# Fabricated citations: numbers outside 1..n_docs, any citation for closed-book.
def test_fabricated_citations():
    # Cites documents 1 and 2 out of 2 shown -> valid.
    ok = _pred("1", '{"answer": "yes", "rationale": "", "citations": [1, 2]}', "", n_docs=2)
    # Cites document 3 but only 2 were shown -> fabricated.
    bad = _pred("2", '{"answer": "yes", "rationale": "", "citations": [3]}', "", n_docs=2)
    # Closed-book (0 documents) cites document 1 -> fabricated.
    closed = _pred("3", '{"answer": "yes", "rationale": "", "citations": [1]}', "", n_docs=0)
    # Parse failure mentioning "[9]" -> ignored, not counted.
    failed = _pred("4", "PARSE_FAILURE: junk [9]", None, n_docs=2)
    # Nothing invalid.
    assert fabricated_citations(ok) == []
    # 3 > n_docs = 2.
    assert fabricated_citations(bad) == [3]
    # n_docs = 0, so citation 1 is invalid.
    assert fabricated_citations(closed) == [1]
    # Parse failures are skipped.
    assert fabricated_citations(failed) == []


# End-to-end check of check_claims + summarize using a fake scorer.
def test_check_claims_and_summary_with_stub_scorer():
    # Two predictions: one normal answer with two claims, one parse failure.
    preds = [
        _pred(
            "1",  # question id
            '{"answer": "yes", "rationale": "x", "citations": [1]}',  # valid: cites doc 1 of 1
            "Drug A reduced the symptoms. Drug A is cheap and widely used.",  # 5 + 7 words
            n_docs=1,  # one document shown
        ),
        # Parse failure: skipped by check_claims, counted in parse_failures.
        _pred("2", "PARSE_FAILURE: nothing", None, n_docs=1),
    ]

    def scorer(pairs):  # supported only if the claim mentions "reduced"
        # Return E (entailment) for claims containing "reduced", else N (neutral).
        return [E if "reduced" in claim else N for _, claim in pairs]

    # One evidence premise ("doc text") and two gold premises for every prediction.
    rows = check_claims(preds, lambda p: ["doc text"], lambda p: ["gold a", "gold b"], scorer)
    # Claim 0 mentions "reduced" -> supported twice; claim 1 does not -> not_supported twice.
    assert [(r["claim_index"], r["evidence_label"], r["gold_label"]) for r in rows] == [
        (0, "supported", "supported"),
        (1, "not_supported", "not_supported"),
    ]
    # Summarise the two rows.
    s = summarize(rows, preds)
    # 1 of 2 replies failed; 2 claims; 2 claims / 1 parsed answer = 2 per answer.
    assert s["parse_failures"] == 1 and s["claims"] == 2 and s["claims_per_answer"] == 2
    # 1 of 2 claims supported by gold -> 1/2 = 0.5.
    assert s["gold_grounded"]["supported"] == 0.5
    # The one parsed answer has an unsupported claim -> 1/1 = 1.0.
    assert s["answers_with_claim_not_supported_by_gold"] == 1.0
    # Its citation [1] is valid (1 doc shown) -> 0/1 = 0.0.
    assert s["answers_with_fabricated_citation"] == 0.0


# Closed-book has no documents, so its evidence label must be None (not applicable).
def test_closed_book_has_no_evidence_label():
    # A closed-book prediction with one 8-word claim and 0 documents.
    pred = _pred(
        "1",  # question id
        '{"answer": "no", "rationale": "", "citations": []}',  # no citations
        "Exposure E was not linked to disease F.",  # the single claim
        n_docs=0,  # nothing retrieved
        system="closed_book",  # system name
    )
    # Empty evidence list; one gold premise; scorer always says entailment.
    rows = check_claims([pred], lambda p: [], lambda p: ["gold"], lambda pairs: [E] * len(pairs))
    # Evidence label skipped (None); gold label still computed (supported).
    assert rows[0]["evidence_label"] is None and rows[0]["gold_label"] == "supported"


# The annotation sample must be repeatable and contain equal numbers per system.
def test_stratified_sample_is_balanced_and_deterministic():
    # Two systems "a" and "b", 10 items each, ids like "a:0".
    items = {s: [{"item_id": f"{s}:{i}"} for i in range(10)] for s in ("a", "b")}
    # Draw 6 items: 6 // 2 systems = 3 per system.
    first = stratified_sample(items, n=6, seed=0)
    # Same seed -> identical sample (same items, same order).
    assert first == stratified_sample(items, n=6, seed=0)
    # First character of each id is the system: expect three "a" and three "b".
    assert sorted(it["item_id"][0] for it in first) == ["a", "a", "a", "b", "b", "b"]


# Per-label precision/recall and accuracy of label_report on a tiny example.
def test_label_report():
    # Item 1 agrees (supported); item 2 does not (contradicted vs not_supported).
    report = label_report(["supported", "contradicted"], ["supported", "not_supported"])
    # 1 of 2 items match -> 0.5.
    assert report["accuracy"] == 0.5
    # "supported" predicted once and correct once -> precision 1/1 = 1.0.
    assert report["supported"]["precision"] == 1.0
    # One true "contradicted" item, never predicted -> recall 0/1 = 0.0.
    assert report["contradicted"]["recall"] == 0.0
