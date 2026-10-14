#!/usr/bin/env python
"""
01_download_data.py
TIPTEKNO'26 - Chaos Game Representation of coronavirus genomes.

Downloads a balanced, reproducible set of complete coronavirus genomes from
NCBI (via Entrez) for genus-level classification:
    Alphacoronavirus / Betacoronavirus / Gammacoronavirus / Deltacoronavirus

- Betacoronavirus explicitly EXCLUDES SARS-CoV-2 (txid2697049) so the genus is
  represented by diverse betacoronaviruses instead of the ~3.16M SARS-CoV-2 flood.
- A separate held-out set of SARS-CoV-2 genomes is downloaded to demonstrate
  "novel pathogen" placement (model never sees SARS-CoV-2 during training).

All accession lists are saved so the dataset is byte-for-byte reproducible.
"""
import os, time, random, csv
from Bio import Entrez, SeqIO

Entrez.email = os.environ.get("NCBI_EMAIL", "fucar@firat.edu.tr")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "Data"))
FASTA_DIR = os.path.join(DATA, "fasta")
os.makedirs(FASTA_DIR, exist_ok=True)

SEED = 42
CAP_PER_GENUS = 250
CAP_SARS2 = 100
LEN_MIN, LEN_MAX = 25000, 35000

GENERA = {
    "Alphacoronavirus": 'Alphacoronavirus[Organism] AND "complete genome"[Title] AND %d:%d[SLEN]' % (LEN_MIN, LEN_MAX),
    "Betacoronavirus":  'Betacoronavirus[Organism] AND "complete genome"[Title] AND %d:%d[SLEN] NOT txid2697049[Organism]' % (LEN_MIN, LEN_MAX),
    "Gammacoronavirus": 'Gammacoronavirus[Organism] AND "complete genome"[Title] AND %d:%d[SLEN]' % (LEN_MIN, LEN_MAX),
    "Deltacoronavirus": 'Deltacoronavirus[Organism] AND "complete genome"[Title] AND %d:%d[SLEN]' % (LEN_MIN, LEN_MAX),
}
SARS2_QUERY = 'txid2697049[Organism] AND "complete genome"[Title] AND 29000:30500[SLEN]'


def esearch_ids(term, retmax):
    for _ in range(4):
        try:
            h = Entrez.esearch(db="nucleotide", term=term, retmax=retmax)
            r = Entrez.read(h); h.close()
            return r["IdList"], int(r["Count"])
        except Exception as e:
            print("  esearch retry:", e); time.sleep(3)
    return [], 0


def efetch_fasta(ids, out_path, label, meta_rows):
    """Fetch FASTA for a list of UIDs in batches, write records, collect metadata."""
    written = 0
    seen_seqs = set()
    with open(out_path, "w") as out:
        for i in range(0, len(ids), 100):
            batch = ids[i:i+100]
            for _ in range(4):
                try:
                    h = Entrez.efetch(db="nucleotide", id=",".join(batch),
                                      rettype="fasta", retmode="text")
                    recs = list(SeqIO.parse(h, "fasta")); h.close()
                    break
                except Exception as e:
                    print("  efetch retry:", e); time.sleep(3); recs = []
            for rec in recs:
                seq = str(rec.seq).upper()
                if not (LEN_MIN <= len(seq) <= LEN_MAX + 500):
                    continue
                # drop exact duplicate sequences
                key = hash(seq)
                if key in seen_seqs:
                    continue
                seen_seqs.add(key)
                SeqIO.write(rec, out, "fasta")
                meta_rows.append({
                    "accession": rec.id,
                    "label": label,
                    "organism": rec.description,
                    "length": len(seq),
                })
                written += 1
            time.sleep(0.4)
    return written


def main():
    random.seed(SEED)
    meta_rows = []
    print("=== Genus-level balanced download (cap=%d/genus, seed=%d) ===" % (CAP_PER_GENUS, SEED))
    for genus, term in GENERA.items():
        ids, total = esearch_ids(term, retmax=100000)
        random.shuffle(ids)
        pick = ids[:CAP_PER_GENUS]
        with open(os.path.join(DATA, "accessions_%s.txt" % genus), "w") as f:
            f.write("\n".join(pick))
        out_fasta = os.path.join(FASTA_DIR, "%s.fasta" % genus)
        n = efetch_fasta(pick, out_fasta, genus, meta_rows)
        print("  %-18s available=%-6d picked=%-4d written=%d" % (genus, total, len(pick), n))

    print("=== SARS-CoV-2 held-out set (cap=%d) ===" % CAP_SARS2)
    ids, total = esearch_ids(SARS2_QUERY, retmax=2000)
    random.shuffle(ids)
    pick = ids[:CAP_SARS2]
    with open(os.path.join(DATA, "accessions_SARS-CoV-2_holdout.txt"), "w") as f:
        f.write("\n".join(pick))
    n = efetch_fasta(pick, os.path.join(FASTA_DIR, "SARS-CoV-2_holdout.fasta"),
                     "SARS-CoV-2", meta_rows)
    print("  SARS-CoV-2 holdout available=%d picked=%d written=%d" % (total, len(pick), n))

    with open(os.path.join(DATA, "metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["accession", "label", "organism", "length"])
        w.writeheader(); w.writerows(meta_rows)

    print("\nTotal records written:", len(meta_rows))
    from collections import Counter
    print("Per-class:", dict(Counter(r["label"] for r in meta_rows)))


if __name__ == "__main__":
    main()
