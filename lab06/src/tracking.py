import json
import zipfile
import pandas as pd
import yaml
from mlflow import MlflowClient

from common import DATA, RESULTS, save_csv


def export_tracking(config):
    store = DATA / 'mlruns'
    client = MlflowClient(tracking_uri=store.as_uri())
    experiment = client.get_experiment_by_name(config['tracking']['experiment'])
    identifiers = []
    for filename in ['worker_summary.csv', 'batch_summary.csv', 'schedule_summary.csv', 'accumulation_summary.csv']:
        identifiers.extend(pd.read_csv(RESULTS / filename).run_id.tolist())
    identifiers.append(json.loads((RESULTS / 'runs/resume_check.json').read_text())['run_id'])
    profiles = client.search_runs([experiment.experiment_id], "tags.study = 'profile'", order_by=['attributes.start_time DESC'])
    identifiers.append(profiles[0].info.run_id)
    if len(identifiers) != 19 or len(set(identifiers)) != 19:
        raise AssertionError('Expected 19 distinct MLflow runs')
    rows = []
    archive = RESULTS / 'mlflow_runs.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as output:
        metadata = store / experiment.experiment_id / 'meta.yaml'
        output.write(metadata, metadata.relative_to(store).as_posix())
        for identifier in identifiers:
            run = client.get_run(identifier)
            if run.info.status != 'FINISHED' or not run.data.params or not run.data.metrics:
                raise AssertionError(f'Incomplete MLflow run: {identifier}')
            artifacts = client.list_artifacts(identifier)
            if not artifacts:
                raise AssertionError(f'Missing artifacts: {identifier}')
            directory = store / experiment.experiment_id / identifier
            for path in sorted(directory.rglob('*')):
                if path.is_file():
                    output.write(path, path.relative_to(store).as_posix())
            rows.append(dict(run_id=identifier, name=run.data.tags['run_name'], study=run.data.tags['study'],
                             status=run.info.status, parameters=len(run.data.params), metrics=len(run.data.metrics),
                             top_level_artifacts=len(artifacts), start_time_ms=run.info.start_time,
                             end_time_ms=run.info.end_time))
    save_csv(RESULTS / 'mlflow_runs.csv', rows)
    print(f'Exported {len(rows)} genuine MLflow runs to {archive.name}.', flush=True)


def restore_tracking():
    target = DATA / 'restored_mlruns'
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(RESULTS / 'mlflow_runs.zip') as archive:
        for item in archive.infolist():
            resolved = (target / item.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise ValueError('Archive contains a path outside the tracking store')
        archive.extractall(target)
    for experiment in target.iterdir():
        if not experiment.is_dir():
            continue
        metadata = experiment / 'meta.yaml'
        if not metadata.exists():
            continue
        values = yaml.safe_load(metadata.read_text())
        values['artifact_location'] = experiment.as_uri()
        metadata.write_text(yaml.safe_dump(values), encoding='utf-8')
        for run in experiment.iterdir():
            metadata = run / 'meta.yaml'
            if not metadata.is_file():
                continue
            values = yaml.safe_load(metadata.read_text())
            values['artifact_uri'] = (run / 'artifacts').as_uri()
            metadata.write_text(yaml.safe_dump(values), encoding='utf-8')
    print(f'mlflow ui --backend-store-uri "{target.as_uri()}"', flush=True)
    return target
