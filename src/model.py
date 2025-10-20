from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score, f1_score
import joblib


@dataclass
class TrainedModel:
    model: OneVsRestClassifier
    classes_: List[str]


def train_one_vs_rest_logistic(
    X: np.ndarray,
    Y: np.ndarray,
    C: float = 1.0,
    max_iter: int = 200,
) -> TrainedModel:
    """Entraîne une régression logistique One-vs-Rest et renvoie le modèle avec les classes."""
    base = LogisticRegression(
        C=C,
        penalty="l2",
        solver="liblinear",  # stable sur CPU
        max_iter=max_iter,
    )
    ovr = OneVsRestClassifier(base)
    ovr.fit(X, Y)
    # Les classes sont les colonnes de Y, ordonnées comme le MultiLabelBinarizer
    # On ne les connaît qu'à l'extérieur; le pipeline doit fournir la liste
    return TrainedModel(model=ovr, classes_=[])


def predict_proba(trained: TrainedModel, X: np.ndarray) -> np.ndarray:
    """Prédit des probabilités par classe (shape: [N, n_classes])."""
    # LogisticRegression de scikit-learn fournit predict_proba pour OVR
    return trained.model.predict_proba(X)


def compute_fmax(Y_true: np.ndarray, Y_scores: np.ndarray, thresholds: np.ndarray | None = None) -> Tuple[float, float]:
    """Calcule Fmax (max F1 sur seuils) et le meilleur seuil.

    - F1 micro (micro-averaged) sur multi-label.
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)
    best_f1 = 0.0
    best_t = 0.0
    for t in thresholds:
        Y_pred = (Y_scores >= t).astype(int)
        f1 = f1_score(Y_true.ravel(), Y_pred.ravel(), average="binary")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_f1), float(best_t)


def compute_aupr(Y_true: np.ndarray, Y_scores: np.ndarray) -> float:
    """Calcule l'Average Precision (auPR) en micro-average."""
    return float(average_precision_score(Y_true.ravel(), Y_scores.ravel()))


def save_model(trained: TrainedModel, path: str | Path, classes: List[str]) -> None:
    """Sauvegarde le modèle et les classes (GO terms)."""
    trained.classes_ = list(classes)
    joblib.dump({"model": trained.model, "classes": trained.classes_}, path)


def load_model(path: str | Path) -> TrainedModel:
    """Recharge le modèle sauvegardé."""
    obj = joblib.load(path)
    tm = TrainedModel(model=obj["model"], classes_=obj["classes"])
    return tm


def generate_submission(
    ids: List[str],
    scores: np.ndarray,
    classes: List[str],
    out_path: str | Path,
    threshold: float = 0.5,
    top_k: int = 50,
) -> None:
    """Génère le fichier de soumission Kaggle au format TSV: id, GO:term, probabilité.

    - threshold: probabilité minimale.
    - top_k: nombre max de termes exportés par protéine (après tri décroissant).
    """
    with open(out_path, "w", encoding="utf-8") as f:
        for pid, row in zip(ids, scores):
            # Trier classes par probas décroissantes
            idx_sorted = np.argsort(row)[::-1]
            exported = 0
            for j in idx_sorted:
                if row[j] < threshold:
                    # Les suivants seront tous plus petits
                    break
                term = classes[j]
                prob = row[j]
                f.write(f"{pid}\t{term}\t{prob:.6f}\n")
                exported += 1
                if exported >= top_k:
                    break