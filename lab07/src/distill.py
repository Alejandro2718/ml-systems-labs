import torch

from common import (ALPHA, LEARNING_RATE, RESULTS, SEED, STUDENT_EPOCHS, TEMPERATURE, eval_loader,
                    evaluate, latency_row, load_data, load_state, save_csv, save_model, setup,
                    size_row, train)
from models import count_macs, parameter_count, student_cnn, teacher_cnn


def main():
    setup()
    train_set, validation, test = load_data()
    teacher = load_state(teacher_cnn(), 'baseline')
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    torch.manual_seed(SEED)
    initial = student_cnn().state_dict()
    rows, histories = [], []
    for label, guide in [('student_plain', None), ('student_distilled', teacher)]:
        model = student_cnn()
        model.load_state_dict(initial)
        history = train(model, train_set, STUDENT_EPOCHS, LEARNING_RATE, label, validation, teacher=guide)
        save_model(model, label)
        macs, _ = count_macs(model)
        row = dict(model=label, teacher='baseline_cnn' if guide else 'none',
                   temperature=TEMPERATURE if guide else '', alpha=ALPHA if guide else '',
                   parameters=parameter_count(model), macs_per_image=macs, epochs=STUDENT_EPOCHS,
                   validation_accuracy=evaluate(model, eval_loader(validation))[1],
                   test_accuracy=evaluate(model, eval_loader(test))[1],
                   training_seconds=sum(h['epoch_seconds'] for h in history))
        row.update(size_row(model))
        row.update(latency_row(model))
        rows.append(row)
        histories += history
        print(f"{label}: test accuracy {row['test_accuracy']:.4f}", flush=True)
    save_csv(RESULTS / 'distillation_summary.csv', rows)
    save_csv(RESULTS / 'distillation_histories.csv', histories)


if __name__ == '__main__':
    main()
