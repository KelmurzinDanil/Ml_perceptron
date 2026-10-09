from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    classification_report,
    confusion_matrix,
)

from perceptron import MLP


root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "dataset" / "data.csv", header=None)

X = df.iloc[:, 2:].to_numpy(dtype=np.float32)
y = df.iloc[:, 1].map({"B": 0, "M": 1}).to_numpy(dtype=np.int64)


train_idx, rest_idx = train_test_split(
    np.arange(len(y)),
    test_size=0.30,
    stratify=y,
    random_state=42,
)

val_idx, test_idx = train_test_split(
    rest_idx,
    test_size=0.50,
    stratify=y[rest_idx],
    random_state=42,
)

dev_idx = np.concatenate([train_idx, val_idx])
X_dev = X[dev_idx]
y_dev = y[dev_idx]

epochs = 100
learning_rate = 0.01
l2 = 0.015
dropout = 0.00
patience = 10


def evaluate(model, X, y):
    logits = np.asarray(
        model.predict_logits(X.tolist()),
        dtype=np.float64,
    )
    assert np.isfinite(logits).all(), "В логитах появились NaN или inf"

    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(
        np.exp(shifted).sum(axis=1, keepdims=True)
    )

    loss = -log_probs[np.arange(len(y)), y].mean()
    predictions = logits.argmax(axis=1)

    return loss, predictions


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

oof_predictions = np.full(len(y_dev), -1, dtype=np.int64)

print(f"Кросс-валидация: {len(y_dev)} объектов")
print(f"За пределами CV: {len(test_idx)} тестовых объектов\n")

for fold, (fit_idx, score_idx) in enumerate(
    cv.split(X_dev, y_dev), start=1
):
    actual_train_idx, stop_idx = train_test_split(
        fit_idx,
        test_size=0.15,
        stratify=y_dev[fit_idx],
        random_state=42,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_dev[actual_train_idx])
    X_stop = scaler.transform(X_dev[stop_idx])
    X_score = scaler.transform(X_dev[score_idx])

    y_train = y_dev[actual_train_idx]
    y_stop = y_dev[stop_idx]
    y_score = y_dev[score_idx]

    model = MLP([30, 16, 8, 2], seed=42, dropout=dropout)
    rng = np.random.default_rng(42)

    best_loss = float("inf")
    best_epoch = 0
    best_model = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))

        model.train_epoch(
            X_train[order].tolist(),
            y_train[order].tolist(),
            learning_rate,
            l2=l2,
        )

        stop_loss, _ = evaluate(model, X_stop, y_stop)

        if stop_loss < best_loss:
            best_loss = stop_loss
            best_epoch = epoch
            best_model = model.snapshot()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    assert best_model is not None
    model = best_model

    score_loss, predictions = evaluate(model, X_score, y_score)
    oof_predictions[score_idx] = predictions

    row = {
        "fold": fold,
        "best_epoch": best_epoch,
        "loss": score_loss,
        "accuracy": accuracy_score(y_score, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_score, predictions
        ),
        "recall_M": recall_score(
            y_score, predictions, pos_label=1, zero_division=0
        ),
        "errors": int(np.sum(y_score != predictions)),
    }
    results.append(row)

    print(
        f"Fold {fold} | best epoch: {best_epoch:3d} | "
        f"loss: {row['loss']:.4f} | "
        f"accuracy: {row['accuracy']:.4f} | "
        f"recall M: {row['recall_M']:.4f} | "
        f"errors: {row['errors']}/{len(y_score)}"
    )


results = pd.DataFrame(results)

print("\nРезультаты по фолдам:")
print(results.round(4).to_string(index=False))

print("\nСреднее ± стандартное отклонение по фолдам:")
for metric in ["loss", "accuracy", "balanced_accuracy", "recall_M"]:
    mean = results[metric].mean()
    std = results[metric].std(ddof=1)
    print(f"{metric:18s}: {mean:.4f} ± {std:.4f}")

assert np.all(oof_predictions >= 0)

print("\nОбщий отчёт по предсказаниям внешних фолдов:")
print(classification_report(
    y_dev,
    oof_predictions,
    labels=[0, 1],
    target_names=["B", "M"],
    digits=4,
    zero_division=0,
))

print("Матрица ошибок: строки — истинный класс, столбцы — прогноз")
print(pd.DataFrame(
    confusion_matrix(y_dev, oof_predictions, labels=[0, 1]),
    index=["B", "M"],
    columns=["B", "M"],
))