from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import CHECKPOINTS, RESULTS, SEED, setup, save_csv
from models import MODEL_NAMES, make_model
from prepare_data import load_data


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    loss_sum = correct = count = 0
    for images, labels in loader:
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    return loss_sum / count, correct / count, correct


def main():
    setup()
    train, validation, test = load_data()
    print("Data ready: 55,000 training / 5,000 validation / 10,000 test", flush=True)
    val_loader = DataLoader(validation, batch_size=128)
    test_loader = DataLoader(test, batch_size=128)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in MODEL_NAMES:
        torch.manual_seed(SEED)
        loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(SEED))
        model = make_model(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, 6):
            model.train()
            loss_sum = correct = seen = 0
            start = perf_counter()
            for images, labels in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(labels)
                correct += (logits.argmax(1) == labels).sum().item()
                seen += len(labels)
            seconds = perf_counter() - start
            assert seen == 55000
            val_loss, val_accuracy, _ = evaluate(model, val_loader)
            print(f"{name} {epoch}/5 loss={loss_sum/seen:.6f} val={val_accuracy:.4f} ({seconds:.1f}s)", flush=True)
        test_loss, test_accuracy, test_correct = evaluate(model, test_loader)
        torch.save(model.state_dict(), CHECKPOINTS / f"{name.lower()}.pth")
        summary.append(dict(model=name, epochs=5, batch_size=128,
                            learning_rate=0.001, test_accuracy=test_accuracy))
        save_csv(RESULTS / "training_summary.csv", summary)
        print(f"{name} test accuracy {test_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
