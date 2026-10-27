import copy
from statistics import median
from time import perf_counter_ns

import torch
from torch import nn
from thop import profile
from torchinfo import summary

from common import CHECKPOINTS, RESULTS, setup, save_csv
from models import MODEL_NAMES, make_model


def count_attention(module, inputs, output):
    query = inputs[0]
    batch, length, width = query.shape
    macs = batch * (4 * length * width * width + 2 * length * length * width)
    module.total_ops += torch.DoubleTensor([macs])


def ignore_ops(module, inputs, output):
    pass


def measured_macs(model, inputs, custom=False):
    rules = None
    if custom:
        rules = {nn.MultiheadAttention: count_attention, nn.LayerNorm: ignore_ops}
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad():
            return profile(copy.deepcopy(model).eval(), inputs=inputs,
                           custom_ops=rules, verbose=True, report_missing=True)[0]
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


def analytical_rows():
    specs = [
        ("MLP", "Linear 784->256", 784 * 256, (784 + 1) * 256),
        ("MLP", "Linear 256->128", 256 * 128, (256 + 1) * 128),
        ("MLP", "Linear 128->10", 128 * 10, (128 + 1) * 10),
        ("CNN", "Conv 1->16 at 28x28", 9 * 1 * 16 * 28 * 28, (9 + 1) * 16),
        ("CNN", "Conv 16->32 at 14x14", 9 * 16 * 32 * 14 * 14, (9 * 16 + 1) * 32),
        ("CNN", "Linear 1568->10", 1568 * 10, (1568 + 1) * 10),
        ("Attention", "Patch projection 49->64 x16", 16 * 49 * 64, (49 + 1) * 64),
    ]
    for i in (1, 2):
        specs.extend([
            ("Attention", f"Encoder {i}: QKV projections", 16 * 64 * 3 * 64, 3 * 64 * (64 + 1)),
            ("Attention", f"Encoder {i}: QK transpose", 16 * 16 * 64, 0),
            ("Attention", f"Encoder {i}: attention times V", 16 * 16 * 64, 0),
            ("Attention", f"Encoder {i}: output projection", 16 * 64 * 64, 64 * (64 + 1)),
            ("Attention", f"Encoder {i}: FFN 64->128", 16 * 64 * 128, 128 * (64 + 1)),
            ("Attention", f"Encoder {i}: FFN 128->64", 16 * 128 * 64, 64 * (128 + 1)),
            ("Attention", f"Encoder {i}: two LayerNorms", 0, 2 * 2 * 64),
        ])
    specs.append(("Attention", "Classifier 64->10", 64 * 10, (64 + 1) * 10))
    return [dict(model=m, layer=l, matrix_macs=c, matrix_flops=2*c, parameters=p)
            for m, l, c, p in specs]


def main():
    setup()
    details = analytical_rows()
    comparison, times = [], []
    inputs = torch.randn(1, 1, 28, 28)
    for name in MODEL_NAMES:
        model = make_model(name).eval()
        model.load_state_dict(torch.load(CHECKPOINTS / f"{name.lower()}.pth", weights_only=True))
        info = summary(copy.deepcopy(model), input_size=(1, 1, 28, 28),
                       col_names=("output_size", "num_params"), mode="eval", verbose=0)
        (RESULTS / f"{name.lower()}_summary.txt").write_text(str(info), encoding="utf-8")
        parameters = sum(row["parameters"] for row in details if row["model"] == name)
        assert info.total_params == sum(p.numel() for p in model.parameters()) == parameters
        default = measured_macs(model, (inputs,))
        adjusted = measured_macs(model, (inputs,), custom=True)
        analytical = sum(row["matrix_flops"] for row in details if row["model"] == name)
        assert int(2 * adjusted) == analytical
        comparison.append(dict(model=name, parameters=parameters,
                               analytical_matrix_flops=analytical,
                               thop_default_ops=int(default),
                               thop_adjusted_matrix_macs=int(adjusted),
                               thop_adjusted_matrix_flops=int(2 * adjusted)))
        for batch, runs in ((1, 100), (128, 20)):
            x = torch.randn(batch, 1, 28, 28)
            with torch.inference_mode():
                model(x)
                measured = []
                for _ in range(runs):
                    start = perf_counter_ns()
                    prediction = model(x)
                    measured.append((perf_counter_ns() - start) / 1e6)
                    del prediction
            times.append(dict(model=name, batch_size=batch, warmups=1, runs=runs,
                              median_batch_ms=median(measured)))
        print(f"{name}: {parameters:,} parameters; {analytical:,} matrix FLOPs", flush=True)

    representatives = [
        ("Linear 784->256", nn.Linear(784, 256), (torch.randn(1, 784),), 2 * 784 * 256),
        ("Conv 1->16 3x3 28x28", nn.Conv2d(1, 16, 3, padding=1), (inputs,), 2 * 9 * 16 * 28 * 28),
    ]
    tokens = torch.randn(1, 16, 64)
    representatives.append(("Self-attention 16 tokens d64 h4", nn.MultiheadAttention(64, 4, batch_first=True),
                            (tokens, tokens, tokens), 2 * (4 * 16 * 64 * 64 + 2 * 16 * 16 * 64)))
    layer_rows = []
    for label, module, values, expected in representatives:
        measured = measured_macs(module, values, custom=True)
        assert int(2 * measured) == expected
        layer_rows.append(dict(layer=label, analytical_matrix_flops=expected,
                               thop_matrix_macs=int(measured), thop_matrix_flops=int(2 * measured)))
    save_csv(RESULTS / "flops_comparison.csv", comparison)
    save_csv(RESULTS / "representative_layers.csv", layer_rows)
    save_csv(RESULTS / "timing_summary.csv", times)


if __name__ == "__main__":
    main()
