from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Set

try:
    import obonet  # type: ignore
    import networkx as nx  # type: ignore
except Exception:
    obonet = None
    nx = None


def load_go_graph(obo_path: str | Path):
    """Charge le graphe GO (go-basic.obo) via obonet. Retourne un networkx graph.

    Si obonet/networkx ne sont pas disponibles, lève une RuntimeError.
    """
    if obonet is None:
        raise RuntimeError("obonet/networkx non installés. Ajoutez 'obonet' et 'networkx' aux dépendances.")
    return obonet.read_obo(str(obo_path))


def valid_terms_set(g) -> Set[str]:
    """Renvoie l'ensemble des IDs de termes valides présents dans le graphe GO."""
    return {n for n in g.nodes if isinstance(n, str) and n.startswith("GO:")}


def build_parent_map(g) -> Dict[str, Set[str]]:
    """Construit un mapping enfant -> parents (is_a et part_of si disponibles).

    obonet crée des arêtes orientées de l'enfant vers le parent pour is_a.
    Les relations 'part_of' sont stockées dans l'attribut 'relationship'.
    """
    parent_map: Dict[str, Set[str]] = {}
    for u, v, data in g.edges(data=True):
        # u = child, v = parent
        if not isinstance(u, str) or not isinstance(v, str):
            continue
        # is_a: généralement sans attribut 'relationship'
        rel = data.get('relationship')
        if rel is None or rel == 'part_of':
            parent_map.setdefault(u, set()).add(v)
    return parent_map


def propagate_scores(scores: Dict[str, float], parent_map: Dict[str, Set[str]], max_passes: int = 5) -> Dict[str, float]:
    """Propage les scores des enfants vers les parents (max des enfants).

    Effectue plusieurs passes jusqu'à convergence ou max_passes.
    """
    updated = dict(scores)
    for _ in range(max_passes):
        changed = False
        for child, parents in parent_map.items():
            sc = updated.get(child, 0.0)
            if sc <= 0.0:
                continue
            for p in parents:
                if updated.get(p, 0.0) < sc:
                    updated[p] = sc
                    changed = True
        if not changed:
            break
    return updated


def filter_to_valid_terms(scores: Dict[str, float], valid: Iterable[str]) -> Dict[str, float]:
    valid_set = set(valid)
    return {t: s for t, s in scores.items() if t in valid_set}