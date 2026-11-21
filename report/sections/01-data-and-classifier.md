# 1. Data and classifier baseline

*Author: Anand Kumar Pandey (Member 1).*

## 1.1 Data

PubMedQA (Jin et al., 2019) asks whether a biomedical research question is answered **yes**,
**no** or **maybe** by a PubMed abstract. We use the 1,000 expert-labelled questions (PQA-L),
downloaded from the official repository at commit `1cbae8e` and checked against pinned SHA-256
checksums.

| Split | Questions | yes | no | maybe |
|---|---|---|---|---|
| train | 400 | 221 | 135 | 44 |
| dev | 100 | 55 | 34 | 11 |
| test (official) | 500 | 276 | 169 | 55 |

The test split is the official 500-question set. The other 500 questions are split 400 / 100
into train and dev, stratified by label with a fixed seed (42); the ID lists are committed
under `data/splits/` and are never regenerated. The test split is protected in code:
`load_split("test")` raises an error unless a command is explicitly given `--allow-test`, which
only the frozen test run in Phase 4 uses. Only the question and the abstract contexts are model
inputs; the article's conclusion (`long_answer`) states the answer and is never used.

## 1.2 Majority baseline

Predicting the most frequent training label (`yes`) for every dev question gives accuracy
0.550 and macro-F1 0.237 (F1 is 0.710 for yes and 0 for no and maybe).

## 1.3 PubMedBERT classifier

Model: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT). Input: the
question and the abstract contexts as a sentence pair, truncated to 512 tokens (only the
contexts are cut). Training: AdamW, learning rate 2e-5, batch size 8, 10 epochs, linear warm-up
over the first 10 % of steps, weight decay 0.01 and gradient clipping at 1.0. Five seeds
(0 to 4) were trained; for each seed the epoch with the best dev macro-F1 was kept (the earlier
epoch on ties). Training ran on a Google Colab T4 GPU. Checkpoints are on the team's shared
Drive folder; their SHA-256 values are in `results/classifier_checkpoints.json`.

## 1.4 Dev results

| System | Accuracy | Macro-F1 | F1 yes | F1 no | F1 maybe |
|---|---|---|---|---|---|
| Majority | 0.550 | 0.237 | 0.710 | 0.000 | 0.000 |
| PubMedBERT (mean ± std, 5 seeds) | 0.576 ± 0.049 | 0.395 ± 0.048 | 0.684 | 0.473 | 0.029 |

Per seed (dev):

| Seed | Best epoch | Accuracy | Macro-F1 |
|---|---|---|---|
| 0 | 9 | 0.640 | 0.441 |
| 1 | 9 | 0.610 | 0.419 |
| 2 | 9 | 0.540 | 0.326 |
| 3 | 2 | 0.520 | 0.366 |
| 4 | 10 | 0.570 | 0.424 |

Standard deviations are sample standard deviations over the five seeds.

The classifier is better than the majority baseline at telling "no" from "yes" (mean F1 for no
0.473 against 0), and its macro-F1 is higher (0.395 against 0.237). Its accuracy is only 2.6
points above the majority baseline on average, and seed 3 is below it. The "maybe" class is
almost never predicted: no seed predicted it for more than 3 of the 100 dev questions, and only
seed 4 got any of them right (F1 0.143); in the other four seeds its F1 is 0. With 400 training examples and only 44 "maybe" questions in the training set, the model
does not learn that class.

## 1.5 Limitations

- Dev scores are optimistic: dev was used to choose the best epoch of each seed. The test result
  in Section 4 is the honest estimate.
- Dev has only 100 questions. The standard error of an accuracy near 0.57 on 100 questions is
  about 0.05, so the 2.6-point gap in accuracy between the classifier and the majority baseline
  is within noise; the macro-F1 gap mainly reflects that the classifier predicts "no" at all.
- The five seeds differ by 12 points of accuracy and about 11 points of macro-F1, so a single
  training run is not reliable and results are reported as mean and standard deviation.
- GPU training is not bit-for-bit deterministic, so retraining gives slightly different numbers.
- Training time per seed was not recorded.
