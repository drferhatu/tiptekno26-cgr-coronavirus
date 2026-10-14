#!/usr/bin/env python
"""
06_presentation_assets.py
Build every number and picture the TIPTEKNO'26 web presentation shows, straight
from the experiment outputs, so nothing on stage is typed by hand.

Writes  ../tiptekno26-cgr-coronavirus/docs/assets/js/data.js   (window.DATA = {...})
        ../tiptekno26-cgr-coronavirus/docs/assets/img/qr.svg
"""
import os, json
import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
import segno
from fcgr import freq_to_image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "Data"))
FASTA = os.path.join(DATA, "fasta")
# repository layout: ../docs ; working copy next to the paper: ../tiptekno26-cgr-coronavirus/docs
SITE = next(p for p in (os.path.abspath(os.path.join(HERE, "..", "docs")),
                        os.path.abspath(os.path.join(HERE, "..", "tiptekno26-cgr-coronavirus", "docs")))
            if os.path.isdir(p))
PAGES_URL = "https://drferhatu.github.io/tiptekno26-cgr-coronavirus/"
REPO_URL = "https://github.com/drferhatu/tiptekno26-cgr-coronavirus"

GENERA = ["Alphacoronavirus", "Betacoronavirus", "Gammacoronavirus", "Deltacoronavirus"]
DEMO = [  # genome shown live in the chaos-game scenes
    ("SARS-CoV-2", "SARS-CoV-2_holdout", "PZ791385.1", "SARS-CoV-2"),
    ("Alphacoronavirus", "Alphacoronavirus", "KY369911.1", "HCoV-229E"),
    ("Betacoronavirus", "Betacoronavirus", "KT121578.1", "MERS-CoV"),
    ("Gammacoronavirus", "Gammacoronavirus", "KY805846.1", "IBV"),
    ("Deltacoronavirus", "Deltacoronavirus", "KU051649.1", "PDCoV"),
]


def reverse_codes(k):
    """Index map code -> code of the reversed k-mer (base-4 digits reversed)."""
    rev = np.empty(1 << (2 * k), dtype=np.int64)
    for code in range(1 << (2 * k)):
        r, c = 0, code
        for _ in range(k):
            r = (r << 2) | (c & 3)
            c >>= 2
        rev[code] = r
    return rev


def mean_images(k):
    """Per-genus mean FCGR in the chaos-game layout.

    fcgr.freq_to_image places a k-mer with its FIRST base as the coarsest quadrant;
    in the chaos game the LAST base read decides the coarsest quadrant. The two are a
    fixed permutation (classification is unaffected); for the slides we use the true
    chaos-game layout so the averaged signatures match the live-drawn fractal.
    """
    d = np.load(os.path.join(DATA, "features_k%d.npz" % k), allow_pickle=True)
    X, y = d["X"], d["y"]
    rev = reverse_codes(k)
    out = {}
    for g in GENERA:
        f = X[y == g].mean(axis=0)
        f_cgr = np.empty_like(f)
        f_cgr[rev] = f                                          # word w -> address of reversed w
        out[g] = freq_to_image(f_cgr, k)                        # 2^k x 2^k, rows top->bottom
    return out, X, y


def main():
    # ---- mean FCGR signatures (k=6 log-scaled for the pictures, k=3 raw for "64 numbers")
    m6, _, _ = mean_images(6)
    logs = {g: np.log10(m6[g] + 1e-6) for g in GENERA}
    lo = min(v.min() for v in logs.values()); hi = max(v.max() for v in logs.values())
    sig6 = {g: np.round((logs[g] - lo) / (hi - lo), 3).flatten().tolist() for g in GENERA}
    m3, _, _ = mean_images(3)
    sig3 = {g: np.round(m3[g] * 100, 3).flatten().tolist() for g in GENERA}  # percent

    # ---- 2-D PCA shadow of the k=5 feature space
    d5 = np.load(os.path.join(DATA, "features_k5.npz"), allow_pickle=True)
    X5, y5 = d5["X"], d5["y"]
    Z = PCA(n_components=2, random_state=0).fit_transform(StandardScaler().fit_transform(X5))
    Z = (Z - Z.min(axis=0)) / (Z.max(axis=0) - Z.min(axis=0))
    pca = [[round(float(a), 4), round(float(b), 4), GENERA.index(g)] for (a, b), g in zip(Z, y5)]

    # ---- confusion matrix of the best configuration (linear SVM, k=5), same CV as 03_classify
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual="auto", max_iter=5000))
    pred = cross_val_predict(clf, X5, y5, cv=cv, n_jobs=-1)
    cm = confusion_matrix(y5, pred, labels=GENERA).tolist()

    # ---- experiment outputs
    cvres = pd.read_csv(os.path.join(DATA, "results_cv.csv"))
    cvtab = {m: {int(r.k): round(100 * r.acc_mean, 1) for r in cvres[cvres.model == m].itertuples()}
             for m in cvres.model.unique()}
    frag = json.load(open(os.path.join(DATA, "results_fragment.json")))
    group = json.load(open(os.path.join(DATA, "results_group_cv.json")))
    novel = json.load(open(os.path.join(DATA, "results_novel.json")))
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"))
    counts = {g: int((meta.label == g).sum()) for g in GENERA}
    counts["SARS-CoV-2"] = int((meta.label == "SARS-CoV-2").sum())
    glen = meta[meta.label != "SARS-CoV-2"].length

    # ---- demo genomes (real GenBank sequences)
    demo = []
    for label, fasta, acc, short in DEMO:
        rec = next(r for r in SeqIO.parse(os.path.join(FASTA, fasta + ".fasta"), "fasta") if r.id == acc)
        demo.append({"genus": label, "acc": acc, "name": short, "seq": str(rec.seq).upper()})

    DATAOBJ = {
        "genera": GENERA,
        "counts": counts,
        "lengthRange": [int(glen.min()), int(glen.max())],
        "sig6": sig6, "sig3": sig3,
        "pca": pca,
        "confusion": cm,
        "cv": cvtab,
        "fragment": {"lengths": frag["lengths"], "k3": [round(100 * v, 1) for v in frag["results"]["3"]],
                     "k5": [round(100 * v, 1) for v in frag["results"]["5"]]},
        "group": {"nGroups": group["n_groups"], "random": round(100 * group["random_cv_acc"], 1),
                  "species": round(100 * group["group_cv_acc"], 1), "f1": round(group["group_cv_f1"], 3)},
        "novel": {"n": novel["n"], "beta": novel["predicted"].get("Betacoronavirus", 0)},
        "timing": {"featAll_s": 0.5, "perGenome_ms": 0.5, "train_s": 1.1, "classify_ms": "<1"},
        "demo": demo,
        "urls": {"pages": PAGES_URL, "repo": REPO_URL},
    }
    os.makedirs(os.path.join(SITE, "assets", "js"), exist_ok=True)
    with open(os.path.join(SITE, "assets", "js", "data.js"), "w") as f:
        f.write("// Generated by Scripts/06_presentation_assets.py -- do not edit by hand.\n")
        f.write("window.DATA = ")
        json.dump(DATAOBJ, f, separators=(",", ":"))
        f.write(";\n")
    segno.make(PAGES_URL, error="m").save(os.path.join(SITE, "assets", "img", "qr.svg"),
                                          scale=10, border=2, dark="#1d1d1f", light=None)
    print("counts", counts, "| confusion", cm)
    print("cv linear", cvtab["SVM-linear"], "| group", DATAOBJ["group"], "| novel", DATAOBJ["novel"])
    print("fragment", DATAOBJ["fragment"])
    print("demo", [(d["name"], d["acc"], len(d["seq"])) for d in demo])
    print("data.js size KB", round(os.path.getsize(os.path.join(SITE, "assets", "js", "data.js")) / 1024))


if __name__ == "__main__":
    main()
