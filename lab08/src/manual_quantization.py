import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from common import RESULTS, eval_loader, evaluate, load_data, load_lab07, models, save_csv, setup

BITS = [8, 4, 2]
TENSOR = '5.weight'


def quantize(weight, bits, per_channel):
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        maximum = weight.abs().flatten(1).amax(dim=1).clamp_min(1e-12)
        scale = (maximum / qmax).view(-1, *([1] * (weight.dim() - 1)))
    else:
        scale = weight.abs().max().clamp_min(1e-12) / qmax
    codes = torch.clamp(torch.round(weight / scale), -qmax, qmax).to(torch.int8)
    return codes, scale, codes.float() * scale


def error_row(weight, bits, per_channel):
    codes, scale, restored = quantize(weight, bits, per_channel)
    error = weight - restored
    mse = float(error.pow(2).mean())
    return dict(bits=bits, granularity='per_channel' if per_channel else 'per_tensor',
                levels_available=2 * (2 ** (bits - 1) - 1) + 1,
                levels_used=int(codes.unique().numel()), zero_point=0,
                scale_min=float(scale.min()), scale_max=float(scale.max()),
                mse=mse, max_abs_error=float(error.abs().max()),
                sqnr_db=float(10 * torch.log10(weight.pow(2).mean() / max(mse, 1e-30))),
                fraction_mapped_to_zero=float((codes == 0).float().mean()),
                int_storage_bits=bits * weight.numel())


def check_against_pytorch(weight):
    codes, scale, restored = quantize(weight, 8, False)
    reference = torch.quantize_per_tensor(weight, float(scale), 0, torch.qint8)
    if not torch.equal(reference.int_repr(), codes):
        raise AssertionError('Manual per-tensor INT8 codes differ from PyTorch')
    codes, scale, restored = quantize(weight, 8, True)
    reference = torch.quantize_per_channel(weight, scale.flatten().double(), torch.zeros(len(scale), dtype=torch.long),
                                           0, torch.qint8)
    if not torch.equal(reference.int_repr(), codes):
        raise AssertionError('Manual per-channel INT8 codes differ from PyTorch')


def model_rows(baseline, test):
    rows = [dict(bits=32, granularity='fp32', test_accuracy=evaluate(baseline, eval_loader(test))[1])]
    for bits in BITS:
        for per_channel in (False, True):
            model = copy.deepcopy(baseline)
            with torch.no_grad():
                for _, module in models.weight_layers(model):
                    module.weight.copy_(quantize(module.weight, bits, per_channel)[2])
            rows.append(dict(bits=bits, granularity='per_channel' if per_channel else 'per_tensor',
                             test_accuracy=evaluate(model, eval_loader(test))[1]))
            print(f"All weights {bits}-bit {rows[-1]['granularity']}: {rows[-1]['test_accuracy']:.4f}", flush=True)
    return rows


def plot(weight, rows):
    plt.rcParams.update({'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.25})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for granularity, marker in [('per_tensor', 'o'), ('per_channel', 's')]:
        part = [r for r in rows if r['granularity'] == granularity]
        axes[0].plot([r['bits'] for r in part], [r['mse'] for r in part], marker + '-', label=granularity)
    axes[0].set(xlabel='Bits', ylabel='Reconstruction MSE', yscale='log', title=f'Baseline {TENSOR}')
    axes[0].set_xticks(BITS)
    axes[0].legend()
    _, _, restored = quantize(weight, 4, False)
    axes[1].hist(weight.flatten().numpy(), bins=120, alpha=0.6, label='FP32 weights')
    axes[1].hist(restored.flatten().numpy(), bins=120, alpha=0.6, label='4-bit per-tensor reconstruction')
    axes[1].set(xlabel='Weight value', ylabel='Count', yscale='log', title='Weight distribution')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / 'manual_quantization.png', dpi=160)
    plt.close(fig)


def main():
    setup()
    baseline = load_lab07('baseline')
    weight = baseline.state_dict()[TENSOR].detach().clone()
    check_against_pytorch(weight)
    rows = [dict(tensor=TENSOR, shape='x'.join(map(str, weight.shape)), **error_row(weight, bits, channel))
            for bits in BITS for channel in (False, True)]
    for row in rows:
        print(f"{row['bits']}-bit {row['granularity']}: MSE={row['mse']:.3e}, SQNR={row['sqnr_db']:.2f} dB", flush=True)
    save_csv(RESULTS / 'manual_quantization.csv', rows)
    plot(weight, rows)
    _, _, test = load_data()
    save_csv(RESULTS / 'manual_quantization_accuracy.csv', model_rows(baseline, test))


if __name__ == '__main__':
    main()
