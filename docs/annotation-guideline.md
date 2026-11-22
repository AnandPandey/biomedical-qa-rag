# Annotation guideline: is a rationale sentence supported?

Each item in `results/annotation/sample.csv` shows one sentence (the **claim**) from an LLM's
rationale and the **premise**: the numbered documents the model was shown or, for closed-book
answers, the sections of the study's own abstract. Decide how the premise relates to the claim
and write one label in the `label` column. Work alone; do not look at the other annotator's
file or at any model output other than the claim.

## Labels

| Label | Use it when |
|---|---|
| `supported` | At least one premise document states the claim, or the claim follows directly from it without outside knowledge |
| `contradicted` | A premise document states the opposite of the claim, or a number or direction that conflicts with it |
| `not_supported` | Neither: the premise does not mention it, or only partly supports it, or the claim relies on outside knowledge |

## Rules for difficult cases

- **Partial support:** if any essential part of the claim is not in the premise, use
  `not_supported`, even if the rest is supported.
- **Hedged claims** ("may", "suggests"): judge the content. "X may reduce Y" is `supported` if
  the premise reports a reduction (even a tentative one).
- **Numbers:** must match the premise (rounding is fine). A different number is `contradicted`.
- **General background knowledge** that the premise does not state ("aspirin is an NSAID") is
  `not_supported`, even if it is true. We measure grounding, not truth.
- **Statements about the documents themselves** ("Document 2 reports...") are judged by
  whether that document really reports it.
- If you cannot decide after re-reading, choose `not_supported` and add a short note in a
  `note` column.

## Worked examples

1. Premise: "Mortality was 12% in the treated group and 20% in controls (p = 0.01)."
   Claim: "Treatment lowered mortality." -> `supported`.
2. Premise: same. Claim: "Treatment lowered mortality by half." -> `contradicted` (12 vs 20 is
   not half).
3. Premise: same. Claim: "The treatment is also cheaper than the alternatives." ->
   `not_supported`.
4. Premise: "No significant difference in pain scores was found." Claim: "Pain improved
   significantly." -> `contradicted`.
5. Premise: "Patients reported fewer headaches; the study was small." Claim: "The drug may
   reduce headaches." -> `supported`.
6. Premise: "Outcomes were measured at 6 months." Claim: "The benefit persisted for years." ->
   `not_supported`.

## Procedure

1. Each annotator copies `sample.csv` to `annotator_a.csv` or `annotator_b.csv` and fills in
   `label` for every row.
2. Run `python -m bmqa.eval.annotate agreement` only after **both** files are complete.
3. Discuss every disagreement together, agree on a label, and record it in
   `adjudicated.csv` (same columns). Do not change your individual files afterwards.
