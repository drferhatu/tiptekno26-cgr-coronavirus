#!/usr/bin/env python
"""
05_group_cv.py
Species-aware cross-validation. Random genome-level CV can overestimate
generalisation when near-identical isolates of the same virus fall in both the
train and test folds. Here we group genomes by species (parsed from the NCBI
organism annotation) and use GroupKFold so that all genomes of a given species
stay on the same side of every split. We compare this stricter estimate against
the random stratified CV for the best configuration (linear SVM, k=5).
"""
import os, re, json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "Data"))
K = 5

_CUT = re.compile(r"\b(strain|isolate|clone|genomic|complete|segment|nonstructural|"
                  r"polyprotein|surface|spike|from|str\.)\b", re.I)


def species_key(desc):
    """Derive a coarse species label from a free-text organism description."""
    s = desc
    # drop leading accession token if present (e.g. 'ON325306.1 ...')
    s = re.sub(r"^\S+\.\d+\s+", "", s)
    m = _CUT.search(s)
    if m:
        s = s[:m.start()]
    s = re.sub(r"[,;].*$", "", s)          # drop trailing clauses
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def main():
    d = np.load(os.path.join(DATA, "features_k%d.npz" % K), allow_pickle=True)
    X, y, acc = d["X"], d["y"], d["acc"]
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"))
    org = dict(zip(meta.accession, meta.organism))
    groups = np.array([species_key(org.get(a, a)) for a in acc])

    n_groups = len(set(groups))
    sizes = pd.Series(groups).value_counts()
    print("Genomes=%d  species-groups=%d  largest group=%d  median=%.0f"
          % (len(y), n_groups, sizes.max(), sizes.median()))

    clf = lambda: make_pipeline(StandardScaler(),
                                LinearSVC(C=1.0, dual="auto", max_iter=5000))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rand_acc = cross_val_score(clf(), X, y, cv=skf, scoring="accuracy")

    gkf = GroupKFold(n_splits=5)
    grp_acc = cross_val_score(clf(), X, y, cv=gkf, groups=groups, scoring="accuracy")
    grp_f1 = cross_val_score(clf(), X, y, cv=gkf, groups=groups, scoring="f1_macro")

    out = {
        "n_groups": int(n_groups), "largest_group": int(sizes.max()),
        "random_cv_acc": float(rand_acc.mean()), "random_cv_std": float(rand_acc.std()),
        "group_cv_acc": float(grp_acc.mean()), "group_cv_std": float(grp_acc.std()),
        "group_cv_f1": float(grp_f1.mean()),
    }
    print("Random  StratifiedKFold acc = %.3f +/- %.3f" % (out["random_cv_acc"], out["random_cv_std"]))
    print("Species GroupKFold      acc = %.3f +/- %.3f  (macro-F1=%.3f)"
          % (out["group_cv_acc"], out["group_cv_std"], out["group_cv_f1"]))
    json.dump(out, open(os.path.join(DATA, "results_group_cv.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
