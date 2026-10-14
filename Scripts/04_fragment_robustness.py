#!/usr/bin/env python
"""
04_fragment_robustness.py
Realistic diagnostic setting: build the reference model from FULL genomes,
then classify SHORT contiguous fragments of held-out genomes.
Reports accuracy vs. fragment length -> shows how far the lightweight FCGR
signature degrades (relevant to partial / low-coverage sequencing & reads).
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from fcgr import encode, kmer_freq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "Data"))
FASTA_DIR = os.path.join(DATA, "fasta")
FIG = os.path.join(HERE, "figures")
LATEX_FIG = os.path.abspath(os.path.join(HERE, "..", "Tiptekno2026_FUEN_LaTeX"))

GENERA = ["Alphacoronavirus", "Betacoronavirus", "Gammacoronavirus", "Deltacoronavirus"]
LENGTHS = [250, 500, 1000, 2000, 5000, 10000]
K_VALUES = [3, 5]
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
RNG = np.random.default_rng(42)


def load():
    seqs, labels = [], []
    for g in GENERA:
        for r in SeqIO.parse(os.path.join(FASTA_DIR, g + ".fasta"), "fasta"):
            seqs.append(str(r.seq)); labels.append(g)
    return seqs, np.array(labels)


def fragment(seq, L, rng):
    if len(seq) <= L:
        return seq
    start = int(rng.integers(0, len(seq) - L + 1))
    return seq[start:start + L]


def main():
    seqs, y = load()
    codes = [encode(s) for s in seqs]
    results = {k: [] for k in K_VALUES}

    for k in K_VALUES:
        Xfull = np.vstack([kmer_freq(c, k) for c in codes]).astype(np.float32)
        # full-genome reference accuracy (upper bound)
        accs_full = []
        for tr, te in CV.split(Xfull, y):
            clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=5000))
            clf.fit(Xfull[tr], y[tr])
            accs_full.append(accuracy_score(y[te], clf.predict(Xfull[te])))
        full_acc = float(np.mean(accs_full))

        for L in LENGTHS:
            # precompute fragment features for this L (seeded, reproducible)
            rng = np.random.default_rng(1000 + L)
            Xfrag = np.vstack([kmer_freq(encode(fragment(s, L, rng)), k)
                               for s in seqs]).astype(np.float32)
            fold_acc = []
            for tr, te in CV.split(Xfull, y):
                clf = make_pipeline(StandardScaler(),
                                    LinearSVC(C=1.0, dual="auto", max_iter=5000))
                clf.fit(Xfull[tr], y[tr])              # train on FULL genomes
                fold_acc.append(accuracy_score(y[te], clf.predict(Xfrag[te])))  # test on fragments
            results[k].append(float(np.mean(fold_acc)))
            print("k=%d  L=%-6d  frag_acc=%.3f" % (k, L, results[k][-1]))
        results[k].append(full_acc)  # full genome as last point
        print("k=%d  L=full   acc=%.3f" % (k, full_acc))

    xs = LENGTHS + [40000]  # plot full genome at a nominal large x
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    for k in K_VALUES:
        ax.plot(xs, results[k], marker="o", label="FCGR $k=%d$" % k)
    ax.set_xscale("log")
    ax.set_xlabel("Query fragment length (bp, log scale; last point = full genome)")
    ax.set_ylabel("Accuracy (train: full, test: fragment)")
    ax.set_ylim(0.2, 1.02); ax.grid(alpha=.3); ax.legend()
    ax.set_title("Robustness of FCGR genus signature to fragment length")
    fig.tight_layout()
    for d in [x for x in (FIG, LATEX_FIG) if os.path.isdir(x)]:
        fig.savefig(os.path.join(d, "fig_fragment.png"), dpi=200)
    plt.close(fig)

    json.dump({"lengths": LENGTHS + ["full"], "results": results},
              open(os.path.join(DATA, "results_fragment.json"), "w"), indent=2)
    print("Saved fragment robustness results & figure.")


if __name__ == "__main__":
    main()
