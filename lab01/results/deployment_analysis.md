# Deployment Budget Analysis

## Measurement basis

Deployment suitability was assessed using the observed inference
peak RSS, median single-sample inference latency, and serialized
model size.

Decimal units are used: 1 KB = 1000 bytes, 1 MB = 1000000 bytes,
and 1 GB = 1000000000 bytes.

| Model | Inference peak RSS (MB) | Median inference (ms) | Model size (bytes) | Model size (KB) |
|---|---:|---:|---:|---:|
| LogisticRegression | 240.951296 | 0.062650 | 1055 | 1.055 |
| RandomForestClassifier | 242.176000 | 3.140700 | 290889 | 290.889 |

## Budget comparison

| Environment | LogisticRegression | RandomForestClassifier |
|---|---|---|
| Cloud | Passes the numerical budget comparison | Passes the numerical budget comparison |
| Edge | Passes with a 256 MB or larger available memory budget | Passes with a 256 MB or larger available memory budget |
| Mobile | Conditional: fits within 256 MB, but not within smaller budgets below its measured requirement | Conditional: fits within 256 MB, but not within smaller budgets below its measured requirement |
| TinyML | Fails the memory budget | Fails both memory and model size budgets |

## Justification

### Cloud

Both models use less than 1 GB of process memory, have measured
inference latency below 100 ms, and occupy less than 500 MB on disk.

### Edge

Both models use less than 256 MB of process memory, have measured
inference latency below 50 ms, and occupy less than 50 MB on disk.

### Mobile

Both models have measured inference latency below 20 ms and occupy
less than 10 MB on disk. However, their process memory requirements
are close to the upper end of the 64-256 MB budget.

At 256 MB, the remaining memory is approximately 15.049 MB for
LogisticRegression and 13.824 MB for RandomForestClassifier.
Additional application components could exceed this margin.

### TinyML

Both models exceed the 256 KB memory ceiling in the measured
Python implementation.

LogisticRegression meets the model size and measured latency limits,
but fails the memory limit.

RandomForestClassifier meets the measured latency limit, but its
290.889 KB model file exceeds the 100 KB size limit. It also fails
the memory limit.

## Limitations

Latency was measured on the host computer, not on Cloud, Edge,
Mobile, or TinyML target hardware. These results provide an initial
budget comparison rather than verified deployment performance.

RSS includes the Python interpreter, imported libraries including
PyTorch, the model, and other live process allocations. It does not
represent the minimum memory footprint of the model alone.

Memory was sampled, so brief peaks may have been missed.
Network delays, application overhead, and model loading were not
included in the inference latency measurement.

The TinyML result applies to this measured implementation.
A different runtime or model representation would require a new
evaluation.
