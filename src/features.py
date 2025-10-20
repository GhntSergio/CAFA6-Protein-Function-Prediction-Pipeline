from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import sparse as sp

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {a: i + 1 for i, a in enumerate(AA_ALPHABET)}  # 1..20, 0=autres


def _stable_hash_kmer(kmer: str, dim: int) -> int:
    """Hash stable et rapide pour un k-mer, modulo dim.

    Utilise un hash polynomial sur AA_TO_INT pour éviter hashlib.
    """
    h = 0
    for c in kmer:
        h = (h * 29 + AA_TO_INT.get(c, 0)) % dim
    return int(h)


def encode_sequence_hashed_counts(seq: str, k: int = 3, dim: int = 1 << 18, normalize: bool = True) -> Dict[int, float]:
    """Encode une séquence en comptage de k-mers hashés dans un espace de dimension 'dim'."""
    s = [c for c in seq if c in AA_TO_INT]
    L = len(s)
    if L < k:
        return {}
    counts: Dict[int, float] = {}
    total = L - k + 1
    for i in range(total):
        idx = _stable_hash_kmer(''.join(s[i:i+k]), dim)
        counts[idx] = counts.get(idx, 0.0) + 1.0
    if normalize and total > 0:
        for kidx in list(counts.keys()):
            counts[kidx] /= float(total)
    return counts


def sequences_to_csr(seqs: List[str], k: int = 3, dim: int = 1 << 18, normalize: bool = True) -> sp.csr_matrix:
    """Encode une liste de séquences en matrice CSR (sparse) de taille [N, dim]."""
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for r, seq in enumerate(seqs):
        counts = encode_sequence_hashed_counts(seq, k=k, dim=dim, normalize=normalize)
        for cidx, val in counts.items():
            rows.append(r)
            cols.append(cidx)
            data.append(val)
    N = len(seqs)
    M = dim
    return sp.csr_matrix((data, (rows, cols)), shape=(N, M), dtype=np.float32)


def iter_csr_rows(seq_iter: Iterable[str], k: int = 3, dim: int = 1 << 18, normalize: bool = True):
    """Générateur de lignes CSR pour encodage streaming de séquences."""
    for seq in seq_iter:
        counts = encode_sequence_hashed_counts(seq, k=k, dim=dim, normalize=normalize)
        if not counts:
            # ligne vide
            yield sp.csr_matrix((1, dim), dtype=np.float32)
        else:
            rows = [0] * len(counts)
            cols = list(counts.keys())
            data = list(counts.values())
            yield sp.csr_matrix((data, (rows, cols)), shape=(1, dim), dtype=np.float32)