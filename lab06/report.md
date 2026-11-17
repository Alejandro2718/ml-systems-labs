# Lab 6: Training Pipelines and Project Proposal

## 1. Goal

I wanted to turn the training code from Lab 3 into a reproducible pipeline and use it to measure where training time goes. I used the CNN, which had the best test accuracy in Lab 3 (0.8927). I profiled one epoch, varied the DataLoader workers and batch size, compared optimizers and learning-rate schedules, checked gradient accumulation and resumed an interrupted run from a checkpoint.

## 2. Method

All settings are in one file, [config.yaml](config.yaml). The training script reads it into a single dictionary, and every experiment changes only a few keys of that dictionary with `--set section.key=value`. The resolved configuration is saved with each MLflow run, so a run can be repeated from its logged parameters.

The CNN is the same as in Lab 3: Conv2d 1->16, ReLU, MaxPool2, Conv2d 16->32, ReLU, MaxPool2, Flatten, Linear 1568->10, with 20,490 parameters. I used the same Fashion-MNIST split as Lab 3: 55,000 training and 5,000 validation images from the official training set, chosen with seed 42, and the official 10,000 test images. Images are converted with ToTensor and Normalize((0.2860,), (0.3530,)). Unlike Lab 3, the data is not cached as tensors, so the DataLoader converts each image while training. This makes data loading part of the measured pipeline.

The base configuration uses Adam, learning rate 0.001, batch size 128, five epochs, cross-entropy on logits and num_workers=0. Python, NumPy and PyTorch use seed 42, deterministic algorithms are enabled and PyTorch uses four intra-op threads and one inter-op thread. The shuffle generator is reseeded with seed + epoch at the start of each epoch, so the batch order of an epoch does not depend on whether the run was restarted.

After every epoch, the script writes a checkpoint with the model, optimizer and scheduler state, the RNG state, the history so far and the MLflow run ID. The file is written to a temporary name first and then renamed, so an interruption during saving does not damage the last checkpoint. Checkpoints are kept in data/lab06_checkpoints, which is not committed.

MLflow uses a local file store in [mlruns](mlruns). Each run logs the flattened configuration as parameters; training loss, training accuracy, validation loss, validation accuracy, learning rate, epoch time and throughput as metrics per epoch; and test accuracy at the end. Artifacts are the resolved config, the history CSV and the final model weights. A resumed run continues the same MLflow run. All runs are exported to [mlflow_runs.csv](results/mlflow_runs.csv), and they can be opened with `mlflow ui --backend-store-uri lab06/mlruns`.

The experiments were:

1. Profiling: one epoch with torch.profiler. Each iteration is split into data_loading (fetching the next batch), forward (model and loss), backward and optimizer_step (step and zero_grad) with record_function. I also timed the same sections with perf_counter, because the profiler adds overhead.
2. Workers: num_workers 0, 1, 2, 4 and 8. For each value I timed one pass through the DataLoader alone and one training epoch, and separated the time the training loop waited for data from compute time. Worker start-up is included.
3. Batch size: 32, 64, 128, 256 and 512, five epochs each with the same Adam learning rate. Each run is a separate process. Peak memory is the peak resident set size of the process, sampled every 5 ms during training. Throughput counts training samples per second of training time, without validation.
4. Schedules and optimizers: SGD with momentum 0.9 (initial lr 0.01) and Adam (initial lr 0.001), each with StepLR (step 4, gamma 0.1), CosineAnnealingLR (T_max 10) and ReduceLROnPlateau on validation loss (factor 0.5, patience 1). Every run had the same 10-epoch budget.
5. Gradient accumulation: batch 256 directly against batch 64 with four accumulation steps, five epochs each. The loss is summed and divided by the number of samples in the accumulation group, so the last, smaller group is weighted the same way as the smaller last batch of the direct run.
6. Resume: the base configuration was stopped after epoch 3 and started again in a new process with `--resume`. I compared its curves with the uninterrupted baseline run.

Run everything from the repository root with `.\.venv\Scripts\python.exe lab06\src\run_lab.py`. One experiment can be repeated with `--steps`, for example `--steps batch report`. A single training run is `.\.venv\Scripts\python.exe lab06\src\train.py --config lab06\config.yaml`.

## 3. Results

Results will be added after the experiments are run.

## 4. Project Proposal

The individual project proposal is in [project/proposal.md](../project/proposal.md).

Sources: supplied Lab 6 manual; Vijay Janapa Reddi, Introduction to Machine Learning Systems, Chapter 8 (sections on training-system architecture, data loading, optimization, profiling and 8.7), Chapter 5 section 5.5 and Chapter 13 section 13.3.
