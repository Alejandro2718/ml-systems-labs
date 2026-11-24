import json
import platform
import random
import warnings
from importlib.metadata import version
from statistics import median
from time import perf_counter, perf_counter_ns

import joblib
import mlflow
import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn

from data_pipeline import DATA, RESULTS, load_config, run

MODEL_FILES = ['vectorizer.joblib', 'model_fp32.pt', 'metadata.json']


def setup(config):
    seed = config['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config['threads'])
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def make_vectorizer(config):
    values = dict(config['vectorizer'])
    values['ngram_range'] = tuple(values['ngram_range'])
    return TfidfVectorizer(dtype=np.float32, **values)


def make_network(inputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def scores(labels, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return dict(threshold=float(threshold), spam_f1=f1_score(labels, predicted, zero_division=0),
                spam_precision=precision_score(labels, predicted, zero_division=0),
                spam_recall=recall_score(labels, predicted, zero_division=0),
                accuracy=accuracy_score(labels, predicted),
                confusion_matrix={'true_ham': matrix[0].tolist(), 'true_spam': matrix[1].tolist()})


def choose_threshold(labels, probabilities, grid):
    candidates = np.round(np.arange(grid[0], grid[1] + 1e-9, grid[2]), 4)
    results = [(f1_score(labels, probabilities >= t, zero_division=0), -abs(t - 0.5), t) for t in candidates]
    return float(max(results)[2])


def train_network(x_train, y_train, x_val, y_val, config):
    values = config['neural']
    torch.manual_seed(config['seed'])
    model = make_network(x_train.shape[1], values['hidden'])
    optimizer = torch.optim.Adam(model.parameters(), lr=values['learning_rate'], weight_decay=values['weight_decay'])
    loss_function = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(config['seed'])
    best, history = None, []
    for epoch in range(1, values['epochs'] + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        total = 0.0
        for batch in order.split(values['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(x_train[batch]).squeeze(1), y_train[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.inference_mode():
            val_loss = loss_function(model(x_val).squeeze(1), y_val).item()
        history.append(dict(epoch=epoch, train_loss=total / len(x_train), validation_loss=val_loss))
        if best is None or val_loss < best[0]:
            best = (val_loss, epoch, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return model.eval(), best[1], history


def pipeline_latency(texts, predict, warmups):
    for text in texts[:warmups]:
        predict(text)
    measured = []
    for text in texts:
        start = perf_counter_ns()
        predict(text)
        measured.append((perf_counter_ns() - start) / 1e6)
    return dict(median_ms=median(measured), p95_ms=float(np.percentile(measured, 95)),
                max_ms=max(measured), messages=len(measured))


def load_artifacts():
    vectorizer = joblib.load(RESULTS / 'vectorizer.joblib')
    metadata = json.loads((RESULTS / 'metadata.json').read_text(encoding='utf-8'))
    model = make_network(metadata['input_features'], metadata['hidden'])
    model.load_state_dict(torch.load(RESULTS / 'model_fp32.pt', weights_only=True))
    return vectorizer, model.eval(), metadata


def main():
    config = load_config()
    setup(config)
    parts, audit = run(config)
    labels = {n: p.label.eq(config['dataset']['positive_label']).astype(int).to_numpy() for n, p in parts.items()}
    texts = {n: p.text.tolist() for n, p in parts.items()}
    vectorizer = make_vectorizer(config)
    matrices = {'train': vectorizer.fit_transform(texts['train'])}
    for name in ('validation', 'test'):
        matrices[name] = vectorizer.transform(texts[name])
    print(f'TF-IDF fitted on training text only: {matrices["train"].shape[1]} features', flush=True)

    logistic = LogisticRegression(random_state=config['seed'], **config['logistic_regression'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        logistic.fit(matrices['train'], labels['train'])
    log_val = logistic.predict_proba(matrices['validation'])[:, 1]
    log_threshold = choose_threshold(labels['validation'], log_val, config['threshold_grid'])

    dense = {n: torch.from_numpy(m.toarray()) for n, m in matrices.items()}
    targets = {n: torch.from_numpy(v).float() for n, v in labels.items()}
    start = perf_counter()
    network, best_epoch, history = train_network(dense['train'], targets['train'], dense['validation'],
                                                 targets['validation'], config)
    training_seconds = perf_counter() - start
    with torch.inference_mode():
        probabilities = {n: torch.sigmoid(network(dense[n]).squeeze(1)).numpy() for n in ('validation', 'test')}
    threshold = choose_threshold(labels['validation'], probabilities['validation'], config['threshold_grid'])

    joblib.dump(vectorizer, RESULTS / 'vectorizer.joblib', compress=3)
    joblib.dump(logistic, RESULTS / 'logistic_regression.joblib', compress=3)
    torch.save(network.state_dict(), RESULTS / 'model_fp32.pt')
    metadata = dict(labels=['ham', 'spam'], positive_label='spam', threshold=threshold,
                    input_features=int(matrices['train'].shape[1]), hidden=config['neural']['hidden'],
                    vectorizer='TF-IDF, fitted on the training split only')
    (RESULTS / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    load_times = []
    for _ in range(5):
        begin = perf_counter()
        loaded_vectorizer, loaded_model, loaded_metadata = load_artifacts()
        load_times.append((perf_counter() - begin) * 1000)
    with torch.inference_mode():
        reloaded = torch.sigmoid(loaded_model(torch.from_numpy(
            loaded_vectorizer.transform(texts['test']).toarray())).squeeze(1)).numpy()
    if not np.allclose(reloaded, probabilities['test'], atol=1e-6):
        raise AssertionError('Reloaded artifacts changed the predictions')

    @torch.inference_mode()
    def predict(text):
        features = torch.from_numpy(loaded_vectorizer.transform([text]).toarray())
        return float(torch.sigmoid(loaded_model(features))[0, 0]) >= loaded_metadata['threshold']

    def predict_logistic(text):
        return bool(logistic.predict_proba(vectorizer.transform([text]))[0, 1] >= log_threshold)

    warmups = config['latency']['warmups']
    artifact_bytes = {name: (RESULTS / name).stat().st_size for name in MODEL_FILES}
    total_bytes = sum(artifact_bytes.values())
    neural = dict(model='TF-IDF + MLP (FP32)', parameters=sum(p.numel() for p in network.parameters()),
                  best_epoch=best_epoch, training_seconds=training_seconds,
                  validation=scores(labels['validation'], probabilities['validation'], threshold),
                  test=scores(labels['test'], probabilities['test'], threshold),
                  latency_batch1=pipeline_latency(texts['test'], predict, warmups),
                  load_ms_median=median(load_times), artifact_bytes=artifact_bytes, total_artifact_bytes=total_bytes)
    baseline = dict(model='TF-IDF + Logistic Regression',
                    validation=scores(labels['validation'], log_val, log_threshold),
                    test=scores(labels['test'], logistic.predict_proba(matrices['test'])[:, 1], log_threshold),
                    latency_batch1=pipeline_latency(texts['test'], predict_logistic, warmups),
                    artifact_bytes=(RESULTS / 'logistic_regression.joblib').stat().st_size
                    + artifact_bytes['vectorizer.joblib'])
    goals = config['targets']
    checks = dict(spam_f1=bool(neural['test']['spam_f1'] >= goals['spam_f1']),
                  p95_latency=bool(neural['latency_batch1']['p95_ms'] <= goals['p95_latency_ms']),
                  artifact_size=bool(total_bytes <= goals['artifact_megabytes'] * 1_000_000))
    metrics = dict(project='Local SMS Spam Classification', milestone='Lab 8 baseline', dataset=audit,
                   environment=dict(python=platform.python_version(), platform=platform.platform(),
                                    torch_threads=config['threads'],
                                    packages={n: version(n) for n in ['torch', 'scikit-learn', 'numpy', 'mlflow']}),
                   models=dict(logistic_regression=baseline, neural_fp32=neural),
                   targets=goals, fp32_targets_met=checks,
                   notes=['Thresholds were selected on validation F1; the test split was evaluated once.',
                          'Latency covers TF-IDF transform, model forward pass and threshold for one message.',
                          'Artifact loading is excluded from latency and reported as load_ms_median.'])

    mlflow.set_tracking_uri((DATA / 'mlruns').as_uri())
    mlflow.set_experiment(config['tracking']['experiment'])
    with mlflow.start_run(run_name='baseline_fp32') as active:
        mlflow.log_params({'seed': config['seed'], 'max_features': config['vectorizer']['max_features'],
                           'ngram_range': str(config['vectorizer']['ngram_range']), 'hidden': config['neural']['hidden'],
                           'epochs': config['neural']['epochs'], 'best_epoch': best_epoch,
                           'learning_rate': config['neural']['learning_rate'], 'threshold': threshold})
        for row in history:
            mlflow.log_metrics({'train_loss': row['train_loss'], 'validation_loss': row['validation_loss']},
                               step=row['epoch'])
        mlflow.log_metrics({'test_spam_f1': neural['test']['spam_f1'], 'validation_spam_f1': neural['validation']['spam_f1'],
                            'p95_latency_ms': neural['latency_batch1']['p95_ms'], 'artifact_bytes': total_bytes})
        metrics['mlflow_run_id'] = active.info.run_id
        (RESULTS / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        for name in MODEL_FILES + ['metrics.json']:
            mlflow.log_artifact(str(RESULTS / name))
    (RESULTS / 'training_history.csv').write_text(
        'epoch,train_loss,validation_loss\n' + ''.join(
            f"{r['epoch']},{r['train_loss']:.6f},{r['validation_loss']:.6f}\n" for r in history), encoding='utf-8')
    print(f"Logistic Regression test F1={baseline['test']['spam_f1']:.4f}; "
          f"MLP test F1={neural['test']['spam_f1']:.4f}, p95={neural['latency_batch1']['p95_ms']:.3f} ms, "
          f"artifacts={total_bytes:,} bytes", flush=True)


if __name__ == '__main__':
    main()
