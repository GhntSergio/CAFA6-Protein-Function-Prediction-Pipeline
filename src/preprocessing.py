from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.preprocessing import MultiLabelBinarizer


# Alphabet standard des acides aminés (20 AA)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_ALPHABET)


def _extract_uniprot_id_train(header: str) -> str:
    """Extrait l'ID UniProt des en-têtes FASTA de train de type 'sp|ID|NAME'."""
    # Exemple: 'sp|A0A0C5B5G6|MOTSC_HUMAN ...'
    parts = header.split("|")
    if len(parts) >= 2:
        return parts[1].strip()
    # Fallback: premier token
    return header.split()[0].strip()


def _extract_uniprot_id_test(header: str) -> str:
    """Extrait l'ID UniProt des en-têtes FASTA de test de type 'ID TAXID'."""
    return header.split()[0].strip()


def load_train_sequences(fasta_path: str | Path) -> pd.DataFrame:
    """Charge les séquences d'entraînement (FASTA) et renvoie DataFrame avec colonnes: id, sequence."""
    records = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        uid = _extract_uniprot_id_train(rec.description)
        seq = str(rec.seq).upper()
        records.append((uid, seq))
    return pd.DataFrame(records, columns=["id", "sequence"])


def load_test_sequences(fasta_path: str | Path) -> pd.DataFrame:
    """Charge les séquences de test (FASTA) avec colonnes: id, sequence."""
    records = []
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        uid = _extract_uniprot_id_test(rec.description)
        seq = str(rec.seq).upper()
        records.append((uid, seq))
    return pd.DataFrame(records, columns=["id", "sequence"])


def load_train_terms(terms_path: str | Path, aspects: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Charge les labels GO en lecture par morceaux pour limiter la mémoire.

    terms.tsv attendu: colonnes [EntryID, term, aspect]. Optionnellement filtre par aspects ("C","F","P").
    Renvoie DataFrame: id, terms (liste).
    """
    aspect_set = set(aspects) if aspects is not None else None
    acc: Dict[str, List[str]] = {}
    for chunk in pd.read_csv(terms_path, sep="\t", chunksize=100_000):
        chunk.rename(columns={"EntryID": "id"}, inplace=True)
        if aspect_set is not None:
            chunk = chunk[chunk["aspect"].isin(aspect_set)]
        for _id, _term in zip(chunk["id"].tolist(), chunk["term"].tolist()):
            if _id not in acc:
                acc[_id] = [_term]
            else:
                acc[_id].append(_term)
    grouped = pd.DataFrame({"id": list(acc.keys()), "terms": list(acc.values())})
    return grouped


def filter_sequences_by_ids(seq_df: pd.DataFrame, valid_ids: Iterable[str]) -> pd.DataFrame:
    """Filtre seq_df pour ne garder que les ids présents dans valid_ids."""
    valid_set = set(valid_ids)
    return seq_df[seq_df["id"].isin(valid_set)].copy()


def build_kmer_index(k: int = 3) -> Dict[str, int]:
    """Construit un index des k-mers sur l'alphabet des AA. Retourne dict kmer->col_index."""
    from itertools import product
    kmers = ["".join(p) for p in product(AA_ALPHABET, repeat=k)]
    return {kmer: i for i, kmer in enumerate(kmers)}


def encode_kmers(seqs: List[str], kmer_index: Dict[str, int], k: int = 3) -> np.ndarray:
    """Encode une liste de séquences en vecteurs de comptage de k-mers normalisés.

    - Renvoie un tableau numpy (float32) de dimension [N, F] où F=len(kmer_index).
    - Normalisation par nombre total de k-mers par séquence.
    """
    n = len(seqs)
    fdim = len(kmer_index)
    X = np.zeros((n, fdim), dtype=np.float32)
    for row, seq in enumerate(seqs):
        # Nettoyage minimal: garder seulement AA connus
        clean_seq = "".join([c for c in seq if c in AA_SET])
        L = len(clean_seq)
        if L < k:
            continue
        total = L - k + 1
        for i in range(total):
            kmer = clean_seq[i : i + k]
            idx = kmer_index.get(kmer)
            if idx is not None:
                X[row, idx] += 1.0
        if total > 0:
            X[row, :] /= float(total)
    return X


def build_label_binarizer(train_labels: pd.DataFrame, min_freq: int = 50) -> Tuple[MultiLabelBinarizer, List[str]]:
    """Construit un MultiLabelBinarizer pour les GO terms, en filtrant les classes rares.

    - min_freq: nombre minimal d'occurrences par terme pour être conservé.
    - Renvoie (mlb, kept_classes)
    """
    # Compter la fréquence des termes
    term_counts: Dict[str, int] = {}
    for terms in train_labels["terms"]:
        for t in terms:
            term_counts[t] = term_counts.get(t, 0) + 1

    kept = {t for t, c in term_counts.items() if c >= min_freq}

    # Filtrer les termes par id
    filtered_terms = train_labels["terms"].apply(lambda lst: [t for t in lst if t in kept])

    mlb = MultiLabelBinarizer(sparse_output=False)
    Y = mlb.fit_transform(filtered_terms)
    kept_classes = list(mlb.classes_)
    return mlb, kept_classes


def train_val_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Division aléatoire train/val sur les identifiants. Conserve distributions globales grossièrement."""
    rng = np.random.default_rng(seed)
    ids = df["id"].unique()
    rng.shuffle(ids)
    n_val = int(len(ids) * val_ratio)
    val_ids = set(ids[:n_val])
    train_df = df[~df["id"].isin(val_ids)].copy()
    val_df = df[df["id"].isin(val_ids)].copy()
    return train_df, val_df


def attach_labels(seq_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Jointure des séquences et des labels sur 'id'. Renvoie DataFrame avec colonnes id, sequence, terms."""
    merged = seq_df.merge(labels_df, on="id", how="inner")
    return merged


def summarize_sequences(seq_df: pd.DataFrame) -> pd.DataFrame:
    """Produit un petit résumé statistique: longueur, proportion d'AA valides, etc."""
    lengths = seq_df["sequence"].str.len()
    valid_prop = seq_df["sequence"].apply(lambda s: sum(1 for c in s if c in AA_SET) / max(1, len(s)))
    summary = pd.DataFrame({
        "count": [len(seq_df)],
        "len_min": [int(lengths.min())],
        "len_max": [int(lengths.max())],
        "len_mean": [float(lengths.mean())],
        "valid_prop_mean": [float(valid_prop.mean())],
    })
    return summary


def prepare_features_and_labels(
    merged_df: pd.DataFrame,
    mlb: MultiLabelBinarizer,
    kmer_index: Dict[str, int],
    k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode X et Y à partir d'un DataFrame contenant id, sequence, terms.

    Renvoie (X, Y, ids)
    """
    seqs = merged_df["sequence"].tolist()
    X = encode_kmers(seqs, kmer_index=kmer_index, k=k)
    Y = mlb.transform(merged_df["terms"])
    ids = merged_df["id"].tolist()
    return X, Y, ids


def prepare_test_features(test_df: pd.DataFrame, kmer_index: Dict[str, int], k: int = 3) -> Tuple[np.ndarray, List[str]]:
    """Encode uniquement les séquences de test en X et renvoie aussi la liste des ids."""
    seqs = test_df["sequence"].tolist()
    X = encode_kmers(seqs, kmer_index=kmer_index, k=k)
    ids = test_df["id"].tolist()
    return X, ids