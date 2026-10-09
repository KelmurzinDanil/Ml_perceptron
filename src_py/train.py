from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, ConfusionMatrixDisplay,
)
from perceptron import MLP


def evaluate(model, X, y):
    logits = np.asarray(model.predict_logits(X.tolist()), dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("В логитах появились NaN или inf")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -log_probs[np.arange(len(y)), y].mean()
    return float(loss), logits.argmax(axis=1)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts"
    output.mkdir(exist_ok=True)

    df = pd.read_csv(root / "dataset/data.csv", header=None)
    if df.empty or df.shape[1] != 32:
        raise ValueError("Ожидается CSV без заголовка: ID, класс и 30 признаков")
    labels = df.iloc[:, 1].map({"B": 0, "M": 1})
    if labels.isna().any():
        raise ValueError("Класс должен быть B или M")
    X = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    y = labels.to_numpy(dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError("В признаках есть пропуски, NaN или inf")

    train_idx, rest_idx = train_test_split(
        np.arange(len(y)), test_size=0.30, stratify=y, random_state=42,
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, stratify=y[rest_idx], random_state=42,
    )

    df.iloc[test_idx].to_csv(output / "test.csv", header=False, index=False)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    print(f"Train: {len(train_idx)}, validation: {len(val_idx)}, test: {len(test_idx)}")
    sizes = [30, 16, 8, 2]
    seed = 42
    dropout = 0.3
    epochs = 100
    learning_rate = 0.01
    l2 = 0.001
    patience = 10

    model = MLP(sizes, seed=seed, dropout=dropout)
    rng = np.random.default_rng(seed)
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_loss = float("inf")
    best_epoch = 0
    best_model = None
    epochs_without_improvement = 0



    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))
        model.train_epoch(
            X_train[order].tolist(), y_train[order].tolist(),
            learning_rate, l2=l2,
        )
        train_loss, train_pred = evaluate(model, X_train, y_train)
        val_loss, val_pred = evaluate(model, X_val, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        train_accuracies.append(accuracy_score(y_train, train_pred))
        val_accuracies.append(accuracy_score(y_val, val_pred))

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = model.snapshot()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d} | loss: {train_loss:.4f} / {val_loss:.4f} | "
            f"val accuracy: {accuracy_score(y_val, val_pred):.4f}"
        )
        if epochs_without_improvement >= patience:
            print("Early stopping")
            break

    if best_model is None:
        raise RuntimeError("Не удалось получить модель с конечной validation loss")
    model = best_model
    val_loss, val_pred = evaluate(model, X_val, y_val)

    sizes, weights, biases, activations, dropout = model.export_state()
    saved = {
        "format_version": 1,
        "model": {
            "sizes": sizes,
            "weights": weights,
            "biases": biases,
            "activations": activations,
            "dropout": dropout,
        },
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "classes": ["B", "M"],
        "training": {
            "seed": seed, "best_epoch": best_epoch,
            "validation_loss": val_loss, "learning_rate": learning_rate,
            "l2": l2, "patience": patience,
        },
    }
    model_path = output / "model.json"
    model_path.write_text(
        json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    loaded = json.loads(model_path.read_text(encoding="utf-8"))
    restored = MLP.from_state(**loaded["model"])
    np.testing.assert_array_equal(
        model.predict_logits(X_val.tolist()),
        restored.predict_logits(X_val.tolist()),
    )
    print(f"\nСохранена эпоха {best_epoch}: {model_path}")
    print("Проверка сохранения: логиты восстановленной модели совпадают")
    print(f"Validation BCE:    {val_loss:.4f}")
    print(f"Accuracy:          {accuracy_score(y_val, val_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_val, val_pred):.4f}")
    print(classification_report(
        y_val, val_pred, labels=[0, 1], target_names=["B", "M"],
        digits=4, zero_division=0,
    ))

    completed_epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(completed_epochs, train_losses, label="Train")
    axes[0].plot(completed_epochs, val_losses, label="Validation")
    axes[0].set_ylabel("Binary cross-entropy")
    axes[0].set_title("Loss")

    axes[1].plot(completed_epochs, train_accuracies, label="Train")
    axes[1].plot(completed_epochs, val_accuracies, label="Validation")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")

    for ax in axes:
        ax.axvline(
            best_epoch,
            color="gray",
            linestyle="--",
            label=f"Best epoch: {best_epoch}",
        )
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(root / "learning_curves.png", dpi=200)

    fig_cm, ax_cm = plt.subplots(figsize=(4, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_val,
        val_pred,
        labels=[0, 1],
        display_labels=["B", "M"],
        ax=ax_cm,
        colorbar=False,
    )

    ax_cm.set_title("Validation: best model")
    fig_cm.tight_layout()
    fig_cm.savefig(root / "validation_confusion.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
