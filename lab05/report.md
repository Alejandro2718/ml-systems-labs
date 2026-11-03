# Lab 5: Frameworks and Automatic Differentiation

## 1. Goal

I wanted to understand what an automatic-differentiation framework does during training. I built a scalar engine, checked its gradients against PyTorch and numerical estimates, and trained the same small network with both implementations. I also compared eager and traced inference and checked how tensor memory layout affects execution time.

## 2. Method

I used Python 3.11.8, PyTorch 2.3.1+cpu and NumPy 1.26.4. All package versions are saved in [versions.txt](results/versions.txt). Python, NumPy, PyTorch and CUDA seeds were set to 42. Measurements used CPU float64 with one intra-op and one inter-op thread.

Each Value node stores its data, gradient, parents and local backward rule. Addition, multiplication, powers and ReLU build the graph during the forward pass. I also implemented exp and log to express sigmoid and binary cross-entropy. Backward first sorts the graph topologically, clears its gradients, sets the output gradient to 1 and visits nodes in reverse order. Local rules use += so a shared value receives contributions from every path. A repeated backward call computes fresh gradients.

For the required expression, I used four elements in each of a, b and c. I summed the four scalar outputs to match ((a*b + c)**2).relu().sum() in PyTorch. Both implementations used float64. Central finite differences used a step of 1e-6. I also checked shared-parent accumulation, repeated backward, ReLU at negative/zero/positive inputs and a zero exponent.

I generated Two Moons with make_moons(n_samples=1000, noise=0.1, random_state=42). A stratified 70/30 split gave 700 training and 300 test samples. The network has 2 inputs, 8 ReLU hidden units and 1 sigmoid output, with 33 trainable parameters. Both engines used identical initial weights, zero biases, the same shuffled batches and SGD with learning rate 0.1. Each fit ran for 50 epochs with batches of 32 and a final batch of 28. BCE was averaged within each batch.

I ran one complete training warm-up per engine and timed five fresh fits. The order of the two engines alternated between runs. Data preparation, initialization, optimizer creation and evaluation were outside the timer. The timer included forward computation, backward computation and parameter updates. Python garbage collection remained enabled during training.

For framework execution, I loaded the trained PyTorch weights, inspected the eager autograd graph and captured the forward graph with torch.jit.trace. Both graphs are saved in [computation_graph.txt](results/computation_graph.txt). Inference used eval and inference_mode in both modes. After correctness checks and one explicit warm-up, I measured 100 calls per mode for batches 1 and 128, alternating the timing order. Trace capture was measured separately.

For memory layout, I sliced every other column of a 512x1024 tensor. The resulting 512x512 view has stride (1024, 2), while its contiguous copy has stride (512, 1). Both contain identical values. I multiplied each by the same 512x64 matrix, and also timed copying the view immediately before multiplication. Each case used one warm-up and 100 measurements. Shapes, dtypes, strides and contiguity are recorded in [tensor_layout.csv](results/tensor_layout.csv).

Run the experiments from the repository root with `.\.venv\Scripts\python.exe lab05\src\run_lab.py`.

## 3. Results

| Gradient comparison | Maximum absolute difference |
|---|---:|
| Scalar engine vs PyTorch | 0.000e+00 |
| Scalar engine vs central differences | 1.684e-10 |

Both differences are below 1e-6. All 12 input gradients passed. The initial model's 33 parameter gradients also matched, with a maximum difference of 2.776e-17. Detailed expression checks are in [gradient_check.csv](results/gradient_check.csv).

| Framework | Median training time, 50 epochs (s) | Final training BCE | Test accuracy |
|---|---:|---:|---:|
| Scalar | 36.0063 | 0.226548 | 92.67% |
| PyTorch | 1.0012 | 0.226548 | 92.67% |

PyTorch was 35.96 times faster for this training workload. Both engines reached the same loss and predicted the same test labels. The largest difference between their final parameters was 1.221e-15. All five fits within each engine reproduced their final weights. The scalar engine constructs many Python objects and executes scalar operations separately; PyTorch processes batches through tensor operations. This comparison includes that implementation difference, not just the backward formula. Raw runs are in [training_times_raw.csv](results/training_times_raw.csv).

| Inference batch | Eager median (ms) | Traced median (ms) | Eager / traced |
|---|---:|---:|---:|
| 1 | 0.034200 | 0.026700 | 1.28x |
| 128 | 0.046700 | 0.039550 | 1.18x |

Tracing took 86.023 ms before the steady-state measurements. Eager and traced outputs matched within 1e-12. The eager autograd graph shows the dependencies used for backward; the traced graph records the forward operations. Both timed modes disabled gradient tracking, so this comparison concerns inference execution overhead. Raw measurements are in [execution_times_raw.csv](results/execution_times_raw.csv).

A reusable graph can reduce repeated Python dispatch and expose operations to optimization. The startup cost has to be recovered through later calls, so tracing or compilation is more useful when a compatible graph is reused often. It does not guarantee faster execution for every model or batch size. Tracing records the executed path; this network has no input-dependent Python branches.

| Tensor-layout experiment | Median time (ms) |
|---|---:|
| Non-contiguous view @ matrix | 2.138200 |
| Pre-copied contiguous tensor @ matrix | 1.821350 |
| Copy view to contiguous, then multiply | 2.141750 |

The non-contiguous case took 1.17 times the pre-copied contiguous case in this experiment. The pre-copied case excludes the cost of creating that copy; the third row includes it. A view can avoid an immediate data copy while changing the access pattern of the next operation. Calling contiguous has its own cost, so the complete operation needs measuring. These results do not establish that every non-contiguous tensor is slower. Raw measurements are in [layout_times_raw.csv](results/layout_times_raw.csv).

## 4. Conclusions

1. Automatic differentiation applies local derivative rules through a recorded graph. The reverse order and gradient accumulation are both necessary when values are reused.
2. Correct gradients do not imply equal execution speed. Both engines learned the same model, while batched tensor operations made PyTorch much faster than scalar Python in this setup.
3. Execution mode and memory layout affect system cost without changing the model's mathematical function. I need to include startup or copying costs when deciding whether an optimization is useful.

Sources: supplied Lab 5 manual; Vijay Janapa Reddi, Introduction to Machine Learning Systems, Chapter 7, sections 7.3, 7.4 and 7.6-7.8.
