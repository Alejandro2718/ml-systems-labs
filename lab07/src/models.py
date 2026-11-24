import torch
from torch import nn

TEACHER_WIDTHS = (32, 64, 128)
TEACHER_HIDDEN = 144
STUDENT_HIDDEN = 60


def teacher_cnn(widths=TEACHER_WIDTHS, hidden=TEACHER_HIDDEN):
    a, b, c = widths
    return nn.Sequential(
        nn.Conv2d(1, a, 3, padding=1), nn.ReLU(),
        nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(b, c, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(c * 49, hidden), nn.ReLU(),
        nn.Linear(hidden, 10))


def student_cnn(hidden=STUDENT_HIDDEN):
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 49, hidden), nn.ReLU(),
        nn.Linear(hidden, 10))


def weight_layers(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))]


def conv_layers(model):
    return [module for module in model if isinstance(module, nn.Conv2d)]


def filter_pruned_teacher(model, ratio):
    convs = conv_layers(model)
    keep = []
    for conv in convs:
        count = conv.out_channels - int(round(conv.out_channels * ratio))
        norms = conv.weight.detach().flatten(1).norm(dim=1)
        keep.append(torch.sort(torch.topk(norms, count).indices).values)
    linear = [module for module in model if isinstance(module, nn.Linear)]
    pruned = teacher_cnn(tuple(len(k) for k in keep), linear[0].out_features)
    new_convs = conv_layers(pruned)
    new_linear = [module for module in pruned if isinstance(module, nn.Linear)]
    with torch.no_grad():
        previous = torch.arange(1)
        for old, new, kept in zip(convs, new_convs, keep):
            new.weight.copy_(old.weight[kept][:, previous])
            new.bias.copy_(old.bias[kept])
            previous = kept
        columns = (previous[:, None] * 49 + torch.arange(49)).flatten()
        new_linear[0].weight.copy_(linear[0].weight[:, columns])
        new_linear[0].bias.copy_(linear[0].bias)
        new_linear[1].load_state_dict(linear[1].state_dict())
    return pruned, [len(k) for k in keep]


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def nonzero_count(model):
    return sum(int(torch.count_nonzero(p)) for p in model.parameters())


def count_macs(model, shape=(1, 1, 28, 28)):
    rows = []

    def hook(module, inputs, output):
        weight = module.weight.detach()
        if isinstance(module, nn.Conv2d):
            per_output = weight[0].numel()
            macs = output.numel() * per_output
        else:
            macs = output.numel() * module.in_features
        density = float(torch.count_nonzero(weight)) / weight.numel()
        rows.append((macs, macs * density))

    handles = [module.register_forward_hook(hook) for _, module in weight_layers(model)]
    with torch.inference_mode():
        model.eval()(torch.zeros(shape))
    for handle in handles:
        handle.remove()
    return int(sum(r[0] for r in rows)), int(round(sum(r[1] for r in rows)))
