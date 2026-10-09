from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
)
from perceptron import MLP


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Прогноз сохранённой Rust-модели")
    parser.add_argument(
        "csv", nargs="?", type=Path, default=root / "artifacts/test.csv",
        help="CSV без заголовка: ID, класс B/M и 30 исходных признаков",
    )
    args = parser.parse_args()

    model_path = root / "artifacts/model.json"
    saved = json.loads(model_path.read_text(encoding="utf-8"))
    if saved["format_version"] != 1 or saved["classes"] != ["B", "M"]:
        raise ValueError("Неподдерживаемый формат модели или порядок классов")
    model = MLP.from_state(**saved["model"])
    if saved["model"]["sizes"][-1] != 2:
        raise ValueError("Для этой задачи нужны два выходных логита")

    scaler = StandardScaler()
    scaler.mean_ = np.asarray(saved["scaler"]["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(saved["scaler"]["scale"], dtype=np.float64)
    scaler.n_features_in_ = saved["model"]["sizes"][0]
    expected = (scaler.n_features_in_,)
    if (
        scaler.mean_.shape != expected or scaler.scale_.shape != expected
        or not np.isfinite(scaler.mean_).all()
        or not np.isfinite(scaler.scale_).all()
        or (scaler.scale_ <= 0).any()
    ):
        raise ValueError("Некорректные параметры стандартизации")

    df = pd.read_csv(args.csv, header=None)
    if df.empty or df.shape[1] != scaler.n_features_in_ + 2:
        raise ValueError("Неверное число столбцов или пустой CSV")
    labels = df.iloc[:, 1].map({"B": 0, "M": 1})
    if labels.isna().any():
        raise ValueError("Для оценки ошибки класс должен быть B или M")
    X = df.iloc[:, 2:].to_numpy(dtype=np.float32)
    y = labels.to_numpy(dtype=np.int64)
    if not np.isfinite(X).all():
        raise ValueError("В признаках есть пропуски, NaN или inf")
    X = scaler.transform(X)

    logits = np.asarray(model.predict_logits(X.tolist()), dtype=np.float64)
    if not np.isfinite(logits).all():
        raise ValueError("В логитах появились NaN или inf")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_probs)
    predictions = logits.argmax(axis=1)

    bce = -np.mean(y * log_probs[:, 1] + (1 - y) * log_probs[:, 0])
    print(f"Загружена модель: {model_path}")
    print(f"Данные: {args.csv}, объектов: {len(y)}")
    print(f"Binary cross-entropy: {bce:.6f}")
    print(f"Accuracy:             {accuracy_score(y, predictions):.4f}")
    print(f"Balanced accuracy:    {balanced_accuracy_score(y, predictions):.4f}")
    print(classification_report(
        y, predictions, labels=[0, 1], target_names=["B", "M"],
        digits=4, zero_division=0,
    ))
    print("Матрица ошибок: строки — истинный класс, столбцы — прогноз")
    print(pd.DataFrame(confusion_matrix(y, predictions, labels=[0, 1]), index=["B", "M"], columns=["B", "M"]))

    result = pd.DataFrame({
        "id": df.iloc[:, 0],
        "actual": df.iloc[:, 1],
        "predicted": np.asarray(saved["classes"])[predictions],
        "p_B": probabilities[:, 0],
        "p_M": probabilities[:, 1],
    })
    output_path = root / "artifacts/predictions.csv"
    result.to_csv(output_path, index=False)
    print(f"Предсказания сохранены: {output_path}")


if __name__ == "__main__":
    main()
