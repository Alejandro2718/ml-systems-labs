# SMS Spam Classifier Model Card

## Model

TF-IDF features (5,000 words and two-word phrases) feed an MLP with one 32-unit ReLU hidden layer and one output logit: 160,065 FP32 parameters. It is trained with binary cross-entropy, Adam with learning rate 0.001, batch 64 and seed 42. The epoch with the lowest validation loss is kept (epoch 23 of 30). The decision threshold, 0.64, maximizes validation spam F1.

A TF-IDF + Logistic Regression model is the simple baseline. Its threshold, 0.31, is selected the same way.

## Intended use

The model flags a message for review on a local CPU. A spam prediction is a recommendation, not proof that a message is fraudulent, and a ham prediction does not prove a message is safe.

## Results on the held-out test split

| Model | Spam F1 | Precision | Recall | Accuracy | Confusion matrix [ham, spam] |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9239 | 0.9884 | 0.8673 | 98.20% | ham [677, 1], spam [13, 85] |
| MLP (FP32) | 0.9519 | 1.0000 | 0.9082 | 98.84% | ham [678, 0], spam [9, 89] |

The MLP missed 9 of 98 spam messages and raised no false alarms on 678 ham messages. Batch-1 latency covers TF-IDF, the forward pass and the threshold on one CPU thread: median 0.81 ms and p95 0.98 ms. The vectorizer, weights and metadata occupy 690,267 bytes; loading them takes about 82 ms and is excluded from latency.

These are results from one seed and one split of 776 test messages; one spam message is about 1 percentage point of recall. Full values are in [metrics.json](results/metrics.json).

## Limitations

See the [data card](data_card.md). New spam styles, other languages and messages unlike the 2011 corpus may be misclassified.
