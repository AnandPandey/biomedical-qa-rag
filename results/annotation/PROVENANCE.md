# Provenance of the annotation files

The pre-declared plan (issue #3) called for two human annotators. That did not happen here.
Both annotation files in this folder were produced by an AI assistant (Claude), not by a human.

| File | Who labelled it | Notes |
|---|---|---|
| `annotator_a.csv` | AI assistant (Claude), pass A | Labelled following `docs/annotation-guideline.md`, blind to every NLI output |
| `annotator_b.csv` | AI assistant (Claude), pass B | A separate fresh run that did not see pass A, also blind to NLI output. **Not a human annotation**, and not by Anand Kumar Pandey despite the file name |
| `adjudicated.csv` | AI assistant (Claude), adjudicator run | The 130 rows where passes A and B agreed keep that label. For the 30 rows where they disagreed, a third fresh AI run chose the final label from the guideline, seeing both labels. **No human took part in the adjudication** |

## What this means for the numbers

- Both passes come from the same model family, so they are not independent raters. The kappa
  between them measures how consistent one model is with itself, not inter-human agreement.
- Any "NLI checker versus adjudicated labels" figure computed from these files is a comparison
  between two AI systems. It is **not** a validation against human annotation, which the plan
  required. The plan's rule (kappa below 0.40 means the hallucination numbers carry a warning)
  cannot be satisfied honestly with these files alone.
- The file names `annotator_a.csv` / `annotator_b.csv` are kept only because
  `python -m bmqa.eval.annotate agreement` reads them by default.

This is a deviation from the pre-declared analysis plan and must be reported as such.
