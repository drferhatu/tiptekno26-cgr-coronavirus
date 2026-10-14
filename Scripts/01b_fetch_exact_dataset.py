#!/usr/bin/env python
"""
01b_fetch_exact_dataset.py
Rebuild the EXACT genome set used in the paper from its released accession lists.

01_download_data.py documents how the balanced sample was originally drawn, but NCBI
keeps growing, so re-running that search today would draw a different sample. This
script instead fetches the 976 genus genomes and the 100 held-out SARS-CoV-2 genomes
by their fixed GenBank accession.version identifiers (Data/metadata.csv, also listed
in Data/accessions/*.txt), in the original order, so 02-05 reproduce every number.

    NCBI_EMAIL=you@example.org python 01b_fetch_exact_dataset.py [--check N]

--check N  only fetch the first N genomes of each class and compare them with the
           FASTA files already present in Data/fasta (no files are written).
"""
import os, sys, time
import pandas as pd
from Bio import Entrez, SeqIO

Entrez.email = os.environ.get("NCBI_EMAIL", "fucar@firat.edu.tr")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "Data"))
FASTA_DIR = os.path.join(DATA, "fasta")
FILE_FOR = {"SARS-CoV-2": "SARS-CoV-2_holdout"}            # every other label -> "<label>.fasta"


def fetch(accs):
    recs = {}
    for i in range(0, len(accs), 100):
        batch = accs[i:i + 100]
        for attempt in range(4):
            try:
                h = Entrez.efetch(db="nucleotide", id=",".join(batch), rettype="fasta", retmode="text")
                for r in SeqIO.parse(h, "fasta"):
                    recs[r.id] = r
                h.close()
                break
            except Exception as e:                           # transient NCBI errors
                print("  retry:", e); time.sleep(3 * (attempt + 1))
        time.sleep(0.4)
    missing = [a for a in accs if a not in recs]
    if missing:
        raise SystemExit("NCBI did not return: %s" % ", ".join(missing[:10]))
    return [recs[a] for a in accs]                           # original order


def main():
    check = int(sys.argv[sys.argv.index("--check") + 1]) if "--check" in sys.argv else 0
    meta = pd.read_csv(os.path.join(DATA, "metadata.csv"))
    os.makedirs(FASTA_DIR, exist_ok=True)
    for label, sub in meta.groupby("label", sort=False):
        accs = sub.accession.tolist()[: check or None]
        recs = fetch(accs)
        path = os.path.join(FASTA_DIR, FILE_FOR.get(label, label) + ".fasta")
        if check:
            local = {r.id: str(r.seq).upper() for r in SeqIO.parse(path, "fasta")}
            same = sum(local.get(r.id) == str(r.seq).upper() for r in recs)
            print("%-18s %d/%d identical to local FASTA" % (label, same, len(recs)))
        else:
            with open(path, "w") as out:
                SeqIO.write(recs, out, "fasta")
            print("%-18s %d genomes -> %s" % (label, len(recs), os.path.relpath(path, DATA)))


if __name__ == "__main__":
    main()
