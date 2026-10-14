# Chaos Game Representation as a Lightweight Genomic Signature for Alignment-Free Classification of Coronaviruses

**Ferhat Uçar · Elifnur Yüksel** — Software Engineering, Fırat University, Elazığ, Türkiye

*2026 Medical Technologies Congress (TIPTEKNO'26), Special Session on Chaos Theory in Medical Technologies, Gaziantep, Türkiye, 15–17 October 2026.*

**▶ Presentation:** <https://drferhatu.github.io/tiptekno26-cgr-coronavirus/> (Turkish by default, press <kbd>L</kbd> for English)

---

The Chaos Game Representation (CGR) turns a DNA sequence into a fractal by an iterated function system; its frequency form (FCGR) is the full *k*-mer spectrum arranged as a 2<sup>k</sup> × 2<sup>k</sup> image. We use FCGR as an alignment-free genomic signature and classify complete coronavirus genomes into the four genera with a plain linear SVM.

| Result | Value |
|---|---|
| Balanced dataset (NCBI, complete genomes) | 976 genomes: *Alpha* 247 · *Beta* 241 · *Gamma* 248 · *Delta* 240 |
| Linear SVM, 5-fold CV | 99.9 % accuracy (k ≥ 4); 99.8 % already at k = 3 (64 features) |
| Species-aware CV (156 species groups, GroupKFold) | 99.7 % accuracy, macro-F1 0.992 |
| Held-out SARS-CoV-2 (never used in training) | 100 / 100 assigned to *Betacoronavirus* |
| Train on full genomes, test on 1 kb fragments | k = 5: 91.4 % · k = 3: 57.1 % |
| Speed (Apple M2 Pro laptop) | ≈ 0.5 ms per genome to featurise, 1.1 s to train, < 1 ms to classify |

## What is in this repository

```
Data/
  accessions/*.txt        exact GenBank accession.version of every genome used (976 + 100)
  metadata.csv            accession, class, organism, length — in the order used by the scripts
  results_*.json|csv      outputs behind every table and figure of the paper
Scripts/
  01_download_data.py         how the balanced sample was originally drawn from NCBI
  01b_fetch_exact_dataset.py  rebuilds the EXACT dataset from the accession lists
  02_build_features.py        FCGR features for k = 3..7 (+ signature figure)
  03_classify.py              5-fold CV of four classifiers, confusion matrix, SARS-CoV-2 test
  04_fragment_robustness.py   train on full genomes, test on 250 bp – 10 kb fragments
  05_group_cv.py              species-aware cross-validation
  06_presentation_assets.py   generates docs/assets/js/data.js and the QR code
  fcgr.py                     CGR / FCGR utilities
  figures/                    the figures of the paper
docs/                     the web presentation (GitHub Pages)
```

The genome sequences themselves are openly archived at NCBI and are **not redistributed** here; only their accession lists are. (The presentation embeds five example genomes — PZ791385.1, KY369911.1, KT121578.1, KY805846.1, KU051649.1 — to play the chaos game live.)

## Reproducing the results

```bash
conda create -n cgr python=3.14 -y && conda activate cgr
pip install -r requirements.txt
cd Scripts
NCBI_EMAIL=you@example.org python 01b_fetch_exact_dataset.py   # exact genomes, original order
python 02_build_features.py
python 03_classify.py
python 04_fragment_robustness.py
python 05_group_cv.py
```

All random choices use fixed seeds (`random_state=42` for the splits). `01_download_data.py` is kept to document the original sampling query; NCBI keeps growing, so re-running it today would draw a different sample — use `01b` to reproduce the paper.

### A note on the FCGR picture layout

`fcgr.freq_to_image` places each *k*-mer with its **first** base as the coarsest quadrant; this is the layout of the signature figure in the paper. In the chaos game itself the **last** base read decides the coarsest quadrant. The two layouts are a fixed permutation of the same *k*-mer counts, so every feature vector and every reported number is identical. The web presentation uses the chaos-game layout (see `06_presentation_assets.py`) so that the averaged signatures line up with the fractal drawn live on stage.

## Presenting from the web page

| Key | Action |
|---|---|
| <kbd>→</kbd> <kbd>Space</kbd> <kbd>PgDn</kbd> (clicker) | next build / next section |
| <kbd>←</kbd> <kbd>PgUp</kbd> | back |
| <kbd>F</kbd> | full screen |
| <kbd>L</kbd> | Turkish / English |
| <kbd>N</kbd> | speaker notes overlay (rehearsal) |
| <kbd>P</kbd> | presenter window with notes and timer (for an extended display) |
| <kbd>B</kbd> | black screen |
| <kbd>?</kbd> | shortcuts |

Speaker notes as a single page, e.g. for a phone: <https://drferhatu.github.io/tiptekno26-cgr-coronavirus/?notes>. Visitors can simply scroll; each section opens fully as it reaches the middle of the screen. The page has no external dependencies and also runs offline from `docs/index.html`.

## Citation

```bibtex
@inproceedings{ucar2026cgr,
  title     = {Chaos Game Representation as a Lightweight Genomic Signature for Alignment-Free Classification of Coronaviruses},
  author    = {U{\c{c}}ar, Ferhat and Y{\"u}ksel, Elifnur},
  booktitle = {2026 Medical Technologies Congress (TIPTEKNO)},
  address   = {Gaziantep, T{\"u}rkiye},
  year      = {2026}
}
```
