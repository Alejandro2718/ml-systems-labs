# Local SMS Spam Classification

Individual course project by Alejandro Jimenez, student ID 70480.

The project classifies public SMS messages as ham or spam and compares a small FP32 neural model with dynamic INT8 quantization. The targets are spam F1 >= 0.90, p95 CPU latency <= 50 ms and total deployed artifacts <= 5 MB.

[One-page project proposal](proposal.pdf). The instructor approved the dataset and scope.

## Lab 8 milestone

The milestone contains the data pipeline, the trained FP32 baseline and [metrics.json](results/metrics.json).

- [data_pipeline.py](src/data_pipeline.py) downloads the UCI archive if it is missing, checks the SHA-256 of the archive and of `SMSSpamCollection`, validates every line, removes exact duplicate messages, audits conflicting labels and creates a seeded, stratified 70/15/15 split. It follows the Lab 4 approach: validate first, split before fitting, and fit preprocessing on the training split only.
- [train_baseline.py](src/train_baseline.py) fits TF-IDF on training text, trains the Logistic Regression baseline and the 5,000-32-1 MLP, selects the decision thresholds on validation F1, evaluates the test split once, measures batch-1 latency of the complete pipeline and logs one MLflow run.

| Model | Test spam F1 | Precision | Recall | Accuracy | p95 latency (ms) | Artifacts (bytes) |
| --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | 0.9239 | 0.9884 | 0.8673 | 98.20% | 0.93 | 85,001 |
| TF-IDF + MLP (FP32) | 0.9519 | 1.0000 | 0.9082 | 98.84% | 0.98 | 690,267 |

The FP32 MLP meets all three proposed targets on this split. The next step is dynamic INT8 quantization of its linear layers and the comparison against the 0.01 F1-loss limit.

## Running

The project uses the Lab 6 environment because it needs MLflow. From the repository root:

```
.\.venv-lab06\Scripts\python.exe project\src\train_baseline.py
```

The raw data and processed splits are written to `data/project/`, which is not committed. [split_assignment.csv](results/split_assignment.csv) records the split of every source row, so the split can be checked without redistributing the text. The MLflow store is written to `data/project/mlruns`; the run ID is in metrics.json.

Dataset: Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), DOI [10.24432/C5CC84](https://doi.org/10.24432/C5CC84).
