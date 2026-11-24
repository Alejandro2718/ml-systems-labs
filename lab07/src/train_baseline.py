import torch

from common import (BASELINE_EPOCHS, LEARNING_RATE, RESULTS, SEED, eval_loader, evaluate,
                    latency_row, load_data, save_csv, save_model, setup, size_row, train)
from models import count_macs, parameter_count, teacher_cnn


def main():
    setup()
    train_set, validation, test = load_data()
    torch.manual_seed(SEED)
    model = teacher_cnn()
    history = train(model, train_set, BASELINE_EPOCHS, LEARNING_RATE, 'baseline', validation)
    save_model(model, 'baseline')
    _, val_accuracy = evaluate(model, eval_loader(validation))
    test_loss, test_accuracy = evaluate(model, eval_loader(test))
    macs, _ = count_macs(model)
    row = dict(model='baseline_cnn', parameters=parameter_count(model), macs_per_image=macs,
               epochs=BASELINE_EPOCHS, validation_accuracy=val_accuracy,
               test_loss=test_loss, test_accuracy=test_accuracy,
               training_seconds=sum(h['epoch_seconds'] for h in history))
    row.update(size_row(model))
    row.update(latency_row(model))
    save_csv(RESULTS / 'baseline_history.csv', history)
    save_csv(RESULTS / 'baseline_summary.csv', [row])
    print(f"Baseline: {row['parameters']:,} parameters, test accuracy {test_accuracy:.4f}", flush=True)


if __name__ == '__main__':
    main()
