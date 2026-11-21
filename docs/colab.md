# Running GPU jobs on Google Colab

Classifier training (Phase 1) and NLI claim checking (Phase 4) are much faster on a GPU. A free
Colab or Kaggle T4 GPU is enough. Everything else runs on a laptop CPU.

## 1. Start a GPU runtime

In Colab: Runtime → Change runtime type → T4 GPU.

## 2. Clone the private repository safely

Create a **fine-grained personal access token** on GitHub (Settings → Developer settings →
Fine-grained tokens) with access to this repository only and the permission
"Contents: Read-only". Give it a short expiry. Then, in a notebook cell:

```python
import getpass
import subprocess

token = getpass.getpass("GitHub token: ")  # typed, never stored in the notebook
url = f"https://x-access-token:{token}@github.com/AnandPandey/biomedical-qa-rag.git"
subprocess.run(["git", "clone", url, "biomedical-qa-rag"], check=True)
del token, url
```

Never paste the token into a cell's text, never print it, and never save or share a notebook
that shows it. Revoke the token on GitHub when you no longer need it.

## 3. Install and prepare data

```bash
%cd biomedical-qa-rag
!pip install -q -e ".[classifier]"
!python -m bmqa.data.download
!python -m bmqa.data.splits
!git status --short     # must print nothing: the split files are reproduced exactly
```

## 4. Run the job

Phase 1 classifier training (about a few minutes per seed on a T4):

```bash
!python -m bmqa.classifier.train --config configs/classifier_pubmedbert.yaml
```

## 5. Bring the results back

Colab storage is deleted when the runtime stops. Before it stops:

- Download the files under `results/` that the job created and commit them from your own
  computer (never commit from Colab with your token).
- Upload checkpoint folders (for example `outputs/classifier/seed0/`) to the team's shared
  Google Drive folder. Mount Drive with `from google.colab import drive;
  drive.mount("/content/drive")` and copy the folders there.
