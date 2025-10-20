from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import scipy.sparse as sp
except Exception:  # fallback si scipy indisponible
    sp = None  # type: ignore


def load_ia_weights(ia_path: str | Path) -> Dict[str, float]:
    """Charge les poids IA par terme GO depuis IA.tsv en mode streaming."""
    weights: Dict[str, float] = {}
    for chunk in pd.read_csv(ia_path, sep="\t", dtype=str, chunksize=200_000):
        term_col = None
        for c in ["term", "GO", "GO_id", "go_id", "id"]:
            if c in chunk.columns:
                term_col = c
                break
        wcol = None
        for c in ["ia", "IA", "ic", "information_accretion", "weight"]:
            if c in chunk.columns:
                wcol = c
                break
        if term_col is None or wcol is None:
            cols = list(chunk.columns)
            if len(cols) >= 2:
                term_col, wcol = cols[0], cols[1]
            else:
                continue
        for t, w in zip(chunk[term_col].tolist(), chunk[wcol].tolist()):
            try:
                weights[str(t)] = float(w)
            except Exception:
                continue
    return weights


def compute_weighted_fmax(
    Y_true,
    Y_scores,
    classes: List[str],
    ia_weights: Dict[str, float],
    thresholds: np.ndarray | None = None,
) -> Tuple[float, float]:
    """Calcule Fmax pondéré IA (micro-average), robuste mémoire.

    Accepte:
    - Y_true: np.ndarray dense [N, C] ou scipy.sparse CSR
    - Y_scores: soit np.ndarray dense [N, C], soit vecteur 1D de scores par classe [C]
      (cas baseline: scores constants par classe sur tous les échantillons)
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 51)
    w = np.array([ia_weights.get(t, 0.0) for t in classes], dtype=np.float64)

    # Nombre d'échantillons
    if sp is not None and sp.isspmatrix(Y_true):
        N, C = Y_true.shape
        pos_counts = np.array(Y_true.sum(axis=0)).ravel()  # positives par classe
    else:
        Y_true = np.asarray(Y_true)
        N, C = Y_true.shape
        pos_counts = Y_true.sum(axis=0)

    actual_pos_w = float(np.dot(pos_counts, w))

    # Cas 1: scores constants par classe (vecteur 1D)
    if np.ndim(Y_scores) == 1:
        class_scores = np.asarray(Y_scores)
        best_f1 = 0.0
        best_t = 0.0
        for t in thresholds:
            active = class_scores >= t
            if not np.any(active) or actual_pos_w == 0.0:
                f1 = 0.0
            else:
                tp_w = float(np.dot(pos_counts[active], w[active]))
                pred_pos_w = float(np.sum(w[active])) * float(N)
                precision_w = 0.0 if pred_pos_w == 0.0 else (tp_w / pred_pos_w)
                recall_w = 0.0 if actual_pos_w == 0.0 else (tp_w / actual_pos_w)
                f1 = 0.0 if (precision_w + recall_w == 0.0) else (2.0 * precision_w * recall_w / (precision_w + recall_w))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        return float(best_f1), float(best_t)

    # Cas 2: matrice 2D (potentiellement volumineuse) — calcul direct
    Y_scores = np.asarray(Y_scores)
    best_f1 = 0.0
    best_t = 0.0
    for t in thresholds:
        Y_pred = (Y_scores >= t).astype(np.float64)
        tp_w = float(np.sum((Y_true * Y_pred) * w))
        pred_pos_w = float(np.sum(Y_pred * w))
        if pred_pos_w == 0.0 or actual_pos_w == 0.0:
            f1 = 0.0
        else:
            precision_w = tp_w / pred_pos_w
            recall_w = tp_w / actual_pos_w
            f1 = 0.0 if (precision_w + recall_w == 0.0) else (2.0 * precision_w * recall_w / (precision_w + recall_w))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return float(best_f1), float(best_t)