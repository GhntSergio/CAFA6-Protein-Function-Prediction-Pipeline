import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# Assurer l'accès au paquet top-level (ajoute le dossier racine au PYTHONPATH)
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    load_train_sequences,
    load_train_terms,
    attach_labels,
    train_val_split,
    build_kmer_index,
    prepare_features_and_labels,
    encode_kmers,
)
from src.model import (
    train_one_vs_rest_logistic,
    predict_proba,
    compute_fmax,
    compute_aupr,
    save_model,
    generate_submission,
)


def main():
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    train_dir = data_dir / "Train"
    test_dir = data_dir / "Test"
    submissions_dir = base / "submissions"
    logs_dir = base / "logs"
    submissions_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    # Fichiers (avec résolution multi-emplacements)
    def resolve_path(*candidates):
        for p in candidates:
            if Path(p).exists():
                return Path(p)
        return None

    train_fasta = resolve_path(train_dir / "train_sequences.fasta", data_dir / "train_sequences.fasta")
    train_terms_tsv = resolve_path(train_dir / "train_terms.tsv", data_dir / "train_terms.tsv")
    test_fasta = resolve_path(test_dir / "testsuperset.fasta", data_dir / "testsuperset.fasta")
    go_obo_path = resolve_path(data_dir / "go-basic.obo", train_dir / "go-basic.obo")
    ia_path = resolve_path(data_dir / "IA.tsv", train_dir / "IA.tsv")

    if train_fasta is None or train_terms_tsv is None or test_fasta is None:
        raise FileNotFoundError("Fichiers critiques manquants: train/test FASTA ou train_terms.tsv non trouvés dans data/Train ou data/")

    model_path = base / "logs" / "ovr_logreg.joblib"
    submission_path = submissions_dir / "submission_final.tsv"

    # Hyperparamètres / options
    aspects = ["C", "F", "P"]  # toutes sous-ontologies
    min_freq = 100  # filtrer plus agressivement les classes rares
    k = 2  # k-mer (réduit pour limiter la mémoire)
    max_train_sequences = int(os.environ.get("MAX_TRAIN_SEQUENCES", "2000"))
    threshold = float(os.environ.get("SUBMIT_THRESHOLD", "0.5"))
    top_k = int(os.environ.get("SUBMIT_TOPK", "50"))

    t0 = time.time()

    # Mode minimal: baseline (fixe ou fréquence) pour environnements très contraints en mémoire
    baseline_mode = os.environ.get("MINIMAL_BASELINE", "fixed")
    if baseline_mode in ("1", "fixed"):
        print("[Minimal] Génération de soumission avec baseline fixe (aucun entraînement).")
        fixed_terms = ["GO:0005515", "GO:0016020", "GO:0005829"]  # exemples de termes fréquents
        fixed_probs = [0.10, 0.09, 0.08]
        from Bio import SeqIO
        with open(submission_path, "w", encoding="utf-8") as outf:
            for rec in SeqIO.parse(str(test_fasta), "fasta"):
                pid = rec.description.split()[0].strip()
                for term, prob in zip(fixed_terms, fixed_probs):
                    outf.write(f"{pid}\t{term}\t{prob:.3f}\n")
        with open(logs_dir / "run_log.txt", "w", encoding="utf-8") as lf:
            lf.write("Mode=MinimalFixedBaseline\n")
            lf.write(f"Terms={','.join(fixed_terms)}\n")
        print(f"Soumission écrite: {submission_path}")
        elapsed = time.time() - t0
        print(f"Terminé en {elapsed/60:.1f} min (baseline fixe)")
        return
    elif baseline_mode == "freq":
        print("[Minimal] Génération de soumission avec baseline fréquence + propagation GO + évaluation IA.")
        # 1) Accumuler les fréquences des termes en lecture par morceaux
        term_counts = {}
        for chunk in pd.read_csv(train_terms_tsv, sep="\t", usecols=["term", "aspect"], chunksize=200_000):
            # Filtrer par aspects demandés
            chunk = chunk[chunk["aspect"].isin(aspects)]
            for term in chunk["term"].tolist():
                term_counts[term] = term_counts.get(term, 0) + 1
        if not term_counts:
            raise RuntimeError("Aucune fréquence de terme calculée depuis train_terms.tsv")

        # 2) Charger labels (agrégés) pour obtenir les classes et faire une validation interne
        labels_df = load_train_terms(train_terms_tsv, aspects=aspects)
        from sklearn.preprocessing import MultiLabelBinarizer
        mlb = MultiLabelBinarizer(sparse_output=True)
        mlb.fit(labels_df["terms"])  # fit sans créer de matrice dense
        classes = list(mlb.classes_)
        # Prior par classe (aligné sur classes)
        counts_vec = np.array([term_counts.get(t, 0) for t in classes], dtype=np.float64)
        class_prior = counts_vec / (counts_vec.sum() + 1e-12)

        # Split train/val pour évaluation
        train_df, val_df = train_val_split(labels_df, val_ratio=0.2, seed=42)
        Y_val = mlb.transform(val_df["terms"])  # CSR sparse [Nv, C]

        # 3) Évaluation IA pondérée (micro-average)
        try:
            from src.metrics import load_ia_weights, compute_weighted_fmax
            ia_weights = load_ia_weights(ia_path)
            fmax_w, best_t_w = compute_weighted_fmax(Y_val, class_prior, classes, ia_weights)
            print(f"- Fmax pondéré IA: {fmax_w:.4f} @ t={best_t_w:.2f}")
            best_t = float(best_t_w)
        except Exception as e:
            print(f"[Warn] Évaluation IA indisponible: {e}")
            # fallback: utiliser Fmax avec poids uniformes (=non pondéré)
            ia_weights_fallback = {t: 1.0 for t in classes}
            fmax, best_t_w = compute_weighted_fmax(Y_val, class_prior, classes, ia_weights_fallback)
            print(f"- Fmax (non pondéré, fallback): {fmax:.4f} @ t={best_t_w:.2f}")
            best_t = float(best_t_w)

        # 4) Propagation GO ascendante (enfants -> parents)
        classes_use = classes
        probs_vec = class_prior.copy()
        try:
            from src.ontology import (
                load_go_graph,
                build_parent_map,
                valid_terms_set,
                propagate_scores,
                filter_to_valid_terms,
            )
            g = load_go_graph(go_obo_path)
            parent_map = build_parent_map(g)
            valid_terms = valid_terms_set(g)
            # Construire dict term->score, propager, filtrer
            prob_dict = {t: float(p) for t, p in zip(classes, class_prior)}
            prob_prop = propagate_scores(prob_dict, parent_map, max_passes=5)
            prob_prop = filter_to_valid_terms(prob_prop, valid_terms)
            classes_use = list(prob_prop.keys())
            probs_vec = np.array([prob_prop[t] for t in classes_use], dtype=np.float64)
            print(f"- Propagation GO: {len(classes)} -> {len(classes_use)} termes (après filtrage)")
        except Exception as e:
            print(f"[Warn] Propagation GO indisponible: {e}")

        # 5) Logs
        with open(logs_dir / "run_log.txt", "w", encoding="utf-8") as lf:
            lf.write("Mode=MinimalFrequencyBaseline+GOPropagation\n")
            lf.write(f"Classes={len(classes)}\n")
            lf.write(f"ClassesAfterPropagation={len(classes_use)}\n")
            lf.write(f"BestThreshold={best_t:.4f}\n")

        # 6) Générer la soumission (streaming test), tri décroissant et filtrage par seuil
        from Bio import SeqIO
        idx_sorted = np.argsort(probs_vec)[::-1]
        with open(submission_path, "w", encoding="utf-8") as outf:
            for rec in SeqIO.parse(str(test_fasta), "fasta"):
                pid = rec.description.split()[0].strip()
                exported = 0
                for j in idx_sorted:
                    p = float(probs_vec[j])
                    if p < best_t:
                        break
                    term = classes_use[j]
                    # Assurer probabilité strictement > 0 et ≤ 1
                    p = min(max(p, 1e-6), 1.0)
                    outf.write(f"{pid}\t{term}\t{p:.3f}\n")
                    exported += 1
                    if exported >= top_k:
                        break
        print(f"Soumission écrite: {submission_path}")
        elapsed = time.time() - t0
        print(f"Terminé en {elapsed/60:.1f} min (baseline fréquence + propagation)")
        return

    print("[1/6] Chargement des labels et sélection d'IDs...")
    labels_df = load_train_terms(train_terms_tsv, aspects=aspects)
    # Comptage global et filtrage des classes rares avant de charger les séquences
    term_counts = {}
    for terms in labels_df["terms"]:
        for t in terms:
            term_counts[t] = term_counts.get(t, 0) + 1
    kept = {t for t, c in term_counts.items() if c >= min_freq}
    labels_df["terms"] = labels_df["terms"].apply(lambda lst: [t for t in lst if t in kept])
    # Enlever les IDs sans aucun terme après filtrage
    labels_df = labels_df[labels_df["terms"].map(len) > 0].copy()
    candidate_ids = labels_df["id"].unique().tolist()
    # Sous-échantillonnage d'IDs pour entraînement
    rng = np.random.default_rng(42)
    if len(candidate_ids) > max_train_sequences:
        candidate_ids = list(rng.choice(candidate_ids, size=max_train_sequences, replace=False))
    print(f"- IDs sélectionnés: {len(candidate_ids)}")

    # Charger uniquement les séquences d'entraînement pour ces IDs
    print("[1b/6] Chargement ciblé des séquences d'entraînement...")
    from Bio import SeqIO
    records = []
    wanted = set(candidate_ids)
    for rec in SeqIO.parse(str(train_fasta), "fasta"):
        uid = rec.description.split("|")[1].strip() if "|" in rec.description else rec.description.split()[0].strip()
        if uid in wanted:
            records.append((uid, str(rec.seq).upper()))
    seq_df = pd.DataFrame(records, columns=["id", "sequence"])
    merged = attach_labels(seq_df, labels_df)
    print(f"- Train samples: {len(merged)}")

    print("[2/6] Préparation des labels et baseline prior...")
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(sparse_output=False)
    # Fit classes sur tout labels_df filtré
    Y_all = mlb.fit_transform(labels_df.set_index("id").loc[seq_df["id"], "terms"].tolist())
    kept_classes = list(mlb.classes_)
    print(f"- Classes conservées: {len(kept_classes)}")

    # Prior de classe (fréquence normalisée)
    class_counts = Y_all.sum(axis=0).astype(np.float64)
    class_prior = class_counts / (class_counts.sum() + 1e-12)

    # Split train/val (sur merged labels uniquement)
    train_df, val_df = train_val_split(merged, val_ratio=0.2, seed=42)
    print(f"- Train: {len(train_df)} | Val: {len(val_df)}")

    print("[3/6] Évaluation baseline sur validation...")
    # Y_true (val) via mlb
    Y_val = mlb.transform(val_df["terms"])
    # Scores constants (prior) pour chaque protéine
    scores_val = np.tile(class_prior, (len(val_df), 1))
    # Fmax/auPR
    fmax, best_t = compute_fmax(Y_val, scores_val)
    aupr = compute_aupr(Y_val, scores_val)
    print(f"- Baseline Prior: Fmax={fmax:.4f} @ t={best_t:.2f} | auPR={aupr:.4f}")

    # Écrire logs
    with open(logs_dir / "run_log.txt", "w", encoding="utf-8") as lf:
        lf.write(f"Samples train={len(train_df)}, val={len(val_df)}\n")
        lf.write(f"Classes={len(kept_classes)}\n")
        lf.write(f"Fmax={fmax:.6f}\n")
        lf.write(f"BestThreshold={best_t:.4f}\n")
        lf.write(f"auPR={aupr:.6f}\n")

    print("[4/4] Prédictions test (baseline prior) et génération de la soumission...")
    from Bio import SeqIO
    batch_size = int(os.environ.get("TEST_BATCH_SIZE", "64"))
    # Streaming pour limiter la mémoire
    with open(submission_path, "w", encoding="utf-8") as outf:
        batch_ids = []
        batch_seqs = []
        for rec in SeqIO.parse(str(test_fasta), "fasta"):
            pid = rec.description.split()[0].strip()
            seq = str(rec.seq).upper()
            batch_ids.append(pid)
            batch_seqs.append(seq)
            if len(batch_seqs) >= batch_size:
                # Baseline: même scores prior pour tout le batch
                Sb = np.tile(class_prior, (len(batch_seqs), 1))
                # écrire une portion
                for pid, row in zip(batch_ids, Sb):
                    idx_sorted = np.argsort(row)[::-1]
                    exported = 0
                    for j in idx_sorted:
                        if row[j] < best_t:
                            break
                        term = kept_classes[j]
                        prob = row[j]
                        outf.write(f"{pid}\t{term}\t{prob:.3f}\n")
                        exported += 1
                        if exported >= top_k:
                            break
                batch_ids = []
                batch_seqs = []
        # Dernière portion
        if batch_seqs:
            Sb = np.tile(class_prior, (len(batch_seqs), 1))
            for pid, row in zip(batch_ids, Sb):
                idx_sorted = np.argsort(row)[::-1]
                exported = 0
                for j in idx_sorted:
                    if row[j] < best_t:
                        break
                    term = kept_classes[j]
                    prob = row[j]
                    outf.write(f"{pid}\t{term}\t{prob:.3f}\n")
                    exported += 1
                    if exported >= top_k:
                        break
    print(f"Soumission écrite: {submission_path}")

    elapsed = time.time() - t0
    print(f"Terminé en {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()