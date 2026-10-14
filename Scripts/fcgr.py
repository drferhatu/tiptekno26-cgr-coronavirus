"""
fcgr.py - Frequency Chaos Game Representation (FCGR) utilities.

CGR (Jeffrey, 1990) maps a DNA sequence into the unit square by iterating a
chaos game over the four corners A/C/G/T. The Frequency CGR of order k
(Deschavanne 1999; Almeida 2001) partitions the square into a 2^k x 2^k grid
whose cells count k-mer occurrences -- i.e. FCGR is exactly the k-mer frequency
spectrum arranged as a fractal image. We use the normalized k-mer frequency
vector (length 4^k) as the feature, and the 2^k x 2^k arrangement for plotting.

Corner convention: A=(0,0), C=(0,1), G=(1,1), T=(1,0)  (x right, y up).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_B = np.full(256, -1, np.int8)
for ch, v in zip(b"ACGT", range(4)):
    _B[ch] = v
for ch, v in zip(b"acgt", range(4)):
    _B[ch] = v


def encode(seq: str) -> np.ndarray:
    """Map a sequence to int8 codes A=0,C=1,G=2,T=3; anything else -> -1."""
    a = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
    return _B[a]


def kmer_freq(codes: np.ndarray, k: int) -> np.ndarray:
    """Normalized k-mer frequency vector of length 4^k (first base = MSB).

    Vectorized rolling window; k-mers spanning any non-ACGT position are skipped.
    """
    size = 1 << (2 * k)
    if codes.shape[0] < k:
        return np.zeros(size, dtype=np.float32)
    valid = codes >= 0
    c = np.where(valid, codes, 0).astype(np.int64)
    W = sliding_window_view(c, k)                          # (L, k)
    ok = sliding_window_view(valid, k).all(axis=1)         # window fully ACGT
    weights = (4 ** np.arange(k - 1, -1, -1)).astype(np.int64)   # first base = MSB
    codes_k = W.dot(weights)[ok]
    counts = np.bincount(codes_k, minlength=size).astype(np.float64)
    s = counts.sum()
    if s > 0:
        counts /= s
    return counts.astype(np.float32)


def code_to_rowcol(k: int):
    """Map each base-4 k-mer code to (row, col) in the 2^k x 2^k FCGR image."""
    side = 1 << k
    rows = np.empty(1 << (2 * k), dtype=np.int32)
    cols = np.empty(1 << (2 * k), dtype=np.int32)
    for code in range(1 << (2 * k)):
        x = y = 0
        for pos in range(k):                       # first base = MSB
            d = (code >> (2 * (k - 1 - pos))) & 3
            xbit = (d >> 1) & 1                     # G,T -> x=1
            ybit = (d ^ (d >> 1)) & 1               # C,G -> y=1  (Gray code)
            x = (x << 1) | xbit
            y = (y << 1) | ybit
        rows[code] = (side - 1 - y)                 # y up -> image row down
        cols[code] = x
    return rows, cols


def freq_to_image(freq: np.ndarray, k: int) -> np.ndarray:
    """Arrange a length-4^k frequency vector into a 2^k x 2^k FCGR image."""
    side = 1 << k
    rows, cols = code_to_rowcol(k)
    img = np.zeros((side, side), dtype=np.float64)
    img[rows, cols] = freq
    return img
