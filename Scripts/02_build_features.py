#!/usr/bin/env python
"""
02_build_features.py
Read genus FASTA files, compute normalized FCGR (k-mer frequency) feature
matrices for several k, and render the per-genus CGR signature figure.
"""
import os, glob, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Bio import SeqIO
from fcgr import encode, kmer_freq, freq_to_image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "Data"))
FASTA_DIR = os.path.join(DATA, "fasta")
FIG = os.path.join(HERE, "figures")
LATEX_FIG = os.path.abspath(os.path.join(HERE, "..", "Tiptekno2026_FUEN_LaTeX"))
os.makedirs(FIG, exist_ok=True)

K_LIST = [3, 4, 5, 6, 7]
GENERA = ["Alphacoronavirus", "Betacoronavirus", "Gammacoronavirus", "Deltacoronavirus"]
SIG_K = 6  # order used for the signature image


def load_sequences():
    seqs, labels, accs = [], [], []
    for genus in GENERA:
        path = os.path.join(FASTA_DIR, genus + ".fasta")
        for rec in SeqIO.parse(path, "fasta"):
            seqs.append(str(rec.seq))
            labels.append(genus)
            accs.append(rec.id)
    return seqs, np.array(labels), np.array(accs)


def main():
    t0 = time.time()
    seqs, labels, accs = load_sequences()
    print("Loaded %d genomes" % len(seqs))
    codes = [encode(s) for s in seqs]

    # feature matrices per k
    for k in K_LIST:
        X = np.vstack([kmer_freq(c, k) for c in codes]).astype(np.float32)
        np.savez_compressed(os.path.join(DATA, "features_k%d.npz" % k),
                            X=X, y=labels, acc=accs)
        print("  k=%d  X=%s  (%.1fs)" % (k, X.shape, time.time() - t0))

    # ---- CGR signature figure: mean FCGR per genus at SIG_K ----
    meanimg = {}
    for genus in GENERA:
        idx = np.where(labels == genus)[0]
        acc = np.zeros((1 << SIG_K, 1 << SIG_K))
        for i in idx:
            acc += freq_to_image(kmer_freq(codes[i], SIG_K), SIG_K)
        meanimg[genus] = acc / len(idx)

    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    for ax, genus in zip(axes, GENERA):
        im = np.log10(meanimg[genus] + 1e-6)
        ax.imshow(im, cmap="inferno", origin="upper")
        ax.set_title(genus.replace("coronavirus", "-CoV"), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Mean FCGR genomic signatures (order k=%d)" % SIG_K, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for d in [x for x in (FIG, LATEX_FIG) if os.path.isdir(x)]:
        fig.savefig(os.path.join(d, "fig_signatures.png"), dpi=200)
    plt.close(fig)
    print("Signature figure saved. Done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
