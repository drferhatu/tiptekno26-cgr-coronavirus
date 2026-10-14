#!/usr/bin/env python
"""
03_classify.py
Genus-level FCGR classification of coronaviruses with classical ML.
- Stratified 5-fold CV over k in {3..7} for SVM(linear), SVM(RBF), RF, kNN.
- Confusion matrix for the best (model,k).
- "Novel pathogen" demo: train without SARS-CoV-2, classify held-out SARS-CoV-2.
Outputs metrics (CSV/JSON) and figures (Scripts/figures + LaTeX folder).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from fcgr import encode, kmer_freq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "Data"))
FASTA_DIR = os.path.join(DATA, "fasta")
FIG = os.path.join(HERE, "figures")
LATEX_FIG = os.path.abspath(os.path.join(HERE, "..", "Tiptekno2026_FUEN_LaTeX"))
os.makedirs(FIG, exist_ok=True)

K_LIST = [3, 4, 5, 6, 7]
GENERA = ["Alphacoronavirus", "Betacoronavirus", "Gammacoronavirus", "Deltacoronavirus"]
SHORT = {g: g.replace("coronavirus", "-CoV") for g in GENERA}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def models():
    return {
        "SVM-linear": make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=5000)),
        "SVM-RBF":    make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")),
        "RandomForest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
        "kNN":        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def main():
    rows = []
    for k in K_LIST:
        d = np.load(os.path.join(DATA, "features_k%d.npz" % k), allow_pickle=True)
        X, y = d["X"], d["y"]
        for name, clf in models().items():
            res = cross_validate(clf, X, y, cv=CV,
                                 scoring=["accuracy", "f1_macro"], n_jobs=-1)
            rows.append(dict(
                model=name, k=k,
                acc_mean=res["test_accuracy"].mean(), acc_std=res["test_accuracy"].std(),
                f1_mean=res["test_f1_macro"].mean(), f1_std=res["test_f1_macro"].std()))
            print("k=%d %-12s acc=%.3f±%.3f f1=%.3f" %
                  (k, name, rows[-1]["acc_mean"], rows[-1]["acc_std"], rows[-1]["f1_mean"]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DATA, "results_cv.csv"), index=False)

    best = df.sort_values("acc_mean", ascending=False).iloc[0]
    best_model, best_k = best["model"], int(best["k"])
    print("\nBEST: %s  k=%d  acc=%.3f  f1=%.3f" %
          (best_model, best_k, best["acc_mean"], best["f1_mean"]))

    # ---- k-effect curve ----
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for name in models():
        sub = df[df.model == name].sort_values("k")
        ax.plot(sub.k, sub.acc_mean, marker="o", label=name)
    ax.set_xlabel("FCGR order $k$"); ax.set_ylabel("5-fold CV accuracy")
    ax.set_xticks(K_LIST); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title("Genus classification accuracy vs. FCGR order")
    fig.tight_layout()
    for dd in [x for x in (FIG, LATEX_FIG) if os.path.isdir(x)]:
        fig.savefig(os.path.join(dd, "fig_kcurve.png"), dpi=200)
    plt.close(fig)

    # ---- confusion matrix for best config ----
    d = np.load(os.path.join(DATA, "features_k%d.npz" % best_k), allow_pickle=True)
    X, y = d["X"], d["y"]
    clf = models()[best_model]
    y_pred = cross_val_predict(clf, X, y, cv=CV, n_jobs=-1)
    cm = confusion_matrix(y, y_pred, labels=GENERA)
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    ConfusionMatrixDisplay(cm, display_labels=[SHORT[g] for g in GENERA]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("%s, k=%d (5-fold CV)" % (best_model, best_k))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    for dd in [x for x in (FIG, LATEX_FIG) if os.path.isdir(x)]:
        fig.savefig(os.path.join(dd, "fig_confusion.png"), dpi=200)
    plt.close(fig)

    # ---- novel-pathogen demo: classify held-out SARS-CoV-2 ----
    clf_full = models()[best_model].fit(X, y)
    s2 = [str(r.seq) for r in SeqIO.parse(os.path.join(FASTA_DIR, "SARS-CoV-2_holdout.fasta"), "fasta")]
    Xs2 = np.vstack([kmer_freq(encode(s), best_k) for s in s2]).astype(np.float32)
    pred = clf_full.predict(Xs2)
    from collections import Counter
    dist = Counter(pred)
    novel = dict(n=len(s2), predicted=dict(dist),
                 beta_fraction=dist.get("Betacoronavirus", 0) / len(s2),
                 best_model=best_model, best_k=best_k)
    json.dump(novel, open(os.path.join(DATA, "results_novel.json"), "w"), indent=2)
    print("Novel-pathogen (SARS-CoV-2) -> ", dict(dist),
          " beta_fraction=%.3f" % novel["beta_fraction"])

    json.dump({"best_model": best_model, "best_k": best_k,
               "best_acc": float(best["acc_mean"]), "best_f1": float(best["f1_mean"])},
              open(os.path.join(DATA, "best_config.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
