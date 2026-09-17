#!/usr/bin/env python3
"""Corrupt the word order of a plain-text pretraining corpus.

For every input line we
  1. tokenize (whitespace-split for EN/TR; per-character for ZH via ``--tokenize chars``),
  2. randomly select a fraction ``--corruption_rate`` of token positions,
  3. for each selected position sample a displacement ``d`` from the chosen
     distribution and a direction ``s in {-1, +1}`` (uniform binary),
  4. assign every token a sort key (unselected tokens keep their original
     index, selected tokens use ``i + s*d``) and stable-sort the tokens by
     that key. This realises all displacements jointly while letting
     unselected tokens drift only as their selected neighbours move past them
     (any single sequence of independent left/right shifts on overlapping
     positions is otherwise ill-defined).
  5. re-join with single spaces and write to the output file.

Distance distributions (``--distribution``)
-------------------------------------------
- ``uniform``   Uniform integer in [1, max_distance].
- ``geometric`` Geometric(p). Small moves likely, mild tail. **Default.**
                Recommended because syntactic dependency lengths in natural
                language are roughly geometrically distributed: most
                relations are local, with an exponentially decaying tail of
                long-range relations. Geometric noise therefore corrupts the
                structure the LM actually relies on (local agreement, head-
                dependent order) much more than uniform noise, which spends
                most of its mass on long jumps that just permute unrelated
                distant material.
- ``poisson``   Poisson(lam) + 1. Concentrated around ``lam``; useful if you
                want a fairly fixed displacement scale.
- ``normal``    |Normal(0, sigma)| rounded up. Smooth, light tails.
- ``zipf``      Zipf(a). Heavy power-law tail – many small swaps with
                occasional very long jumps. Good for stress-testing
                long-range structure.

Why not just uniform?
---------------------
Uniform[1, D] is statistically uniform over *position*, but linguistic
structure is not. With D=20 a uniform sampler spends ~95% of its mass on
swaps of length >= 2 and ~50% on swaps of length >= 10 -- those long swaps
mostly scramble inter-clause material rather than local syntax, so the
corpus looks "globally jumbled" but a lot of local n-gram statistics
survive. A geometric (or Poisson/truncated-normal) sampler instead
concentrates noise at the dependency lengths the model actually needs to
learn, while still occasionally producing long jumps via the tail.

Usage
-----
    python src/pretraining_dataset/shuffle_words.py \\
        --input  ~/Morphological/data/NLLB_10M_subset/NLLB_10M.en \\
        --output_dir ~/Morphological/data/NLLB_10M_subset_shuffled/ \\
        --corruption_rate 0.3 \\
        --distribution geometric \\
        --geom_p 0.5 \\
        --max_distance 20 \\
        --seed 42

Chinese (NLLB ZH is largely unsegmented, so use character tokens)::

    python src/pretraining_dataset/shuffle_words.py \\
        --input  ~/Morphological/data/NLLB_10M_subset/NLLB_10M.zh \\
        --output_dir ~/Morphological/data/NLLB_10M_subset_shuffled/ \\
        --tokenize chars \\
        --corruption_rate 0.3 \\
        --distribution geometric \\
        --seed 42
"""

import argparse
import json
import os
import sys
import time
from typing import Callable, List

import numpy as np


# ── tokenization ─────────────────────────────────────────────────────────────

def tokenize(text: str, mode: str) -> List[str]:
    """Split ``text`` into tokens according to ``mode``.

    - ``whitespace``: ``str.split()`` (EN/TR and space-tokenized corpora).
    - ``chars``: one Unicode character per token (ZH / unsegmented text).
    """
    if mode == "whitespace":
        return text.split()
    if mode == "chars":
        return list(text)
    raise ValueError(f"Unknown tokenize mode: {mode}")


def detokenize(tokens: List[str], mode: str) -> str:
    """Inverse of :func:`tokenize` for writing a line back out."""
    if mode == "whitespace":
        return " ".join(tokens)
    if mode == "chars":
        return "".join(tokens)
    raise ValueError(f"Unknown tokenize mode: {mode}")


# ── distance samplers ────────────────────────────────────────────────────────

def make_distance_sampler(args, rng: np.random.Generator) -> Callable[[int], np.ndarray]:
    """Return a function ``sample(n) -> np.ndarray[int]`` of strictly positive
    displacements clipped to ``[1, args.max_distance]``."""

    kind = args.distribution
    max_d = args.max_distance

    if kind == "uniform":
        def sample(n: int) -> np.ndarray:
            return rng.integers(1, max_d + 1, size=n)

    elif kind == "geometric":
        p = args.geom_p
        if not (0.0 < p <= 1.0):
            raise ValueError("--geom_p must be in (0, 1]")
        def sample(n: int) -> np.ndarray:
            d = rng.geometric(p, size=n)  # already >= 1
            return np.clip(d, 1, max_d)

    elif kind == "poisson":
        lam = args.lam
        if lam <= 0:
            raise ValueError("--lam must be > 0")
        def sample(n: int) -> np.ndarray:
            d = rng.poisson(lam, size=n) + 1  # ensure >= 1
            return np.clip(d, 1, max_d)

    elif kind == "normal":
        sigma = args.sigma
        if sigma <= 0:
            raise ValueError("--sigma must be > 0")
        def sample(n: int) -> np.ndarray:
            d = np.abs(rng.normal(0.0, sigma, size=n))
            d = np.maximum(1, np.round(d).astype(np.int64))
            return np.clip(d, 1, max_d)

    elif kind == "zipf":
        a = args.zipf_a
        if a <= 1.0:
            raise ValueError("--zipf_a must be > 1")
        def sample(n: int) -> np.ndarray:
            # numpy zipf can occasionally return huge values; clip.
            d = rng.zipf(a, size=n)
            return np.clip(d, 1, max_d)

    else:
        raise ValueError(f"Unknown distribution: {kind}")

    return sample


# ── per-line shuffling ───────────────────────────────────────────────────────

def shuffle_tokens(
    tokens: list,
    corruption_rate: float,
    sampler: Callable[[int], np.ndarray],
    rng: np.random.Generator,
) -> list:
    """Return a reordered copy of ``tokens`` with the corruption procedure.

    Empty / single-token lines are returned unchanged.
    """
    n = len(tokens)
    if n < 2 or corruption_rate <= 0.0:
        return tokens

    n_move = int(round(corruption_rate * n))
    if n_move <= 0:
        return tokens
    n_move = min(n_move, n)

    move_idx = rng.choice(n, size=n_move, replace=False)
    distances = sampler(n_move).astype(np.float64)
    signs = rng.choice([-1.0, 1.0], size=n_move)

    # Float keys: unmoved tokens get integer keys i; moved tokens get i + s*d.
    # A tiny jitter on the moved keys breaks ties between collisions of moved
    # tokens (and between a moved token landing exactly on an unmoved index)
    # in a uniform random way rather than always preferring one side.
    keys = np.arange(n, dtype=np.float64)
    jitter = rng.uniform(-0.25, 0.25, size=n_move)
    keys[move_idx] = move_idx.astype(np.float64) + signs * distances + jitter

    order = np.argsort(keys, kind="stable")
    return [tokens[i] for i in order]


# ── file processing ──────────────────────────────────────────────────────────

def process_file(args) -> dict:
    rng = np.random.default_rng(args.seed)
    sampler = make_distance_sampler(args, rng)
    tok_mode = args.tokenize

    in_path = os.path.expanduser(args.input)
    if not os.path.isfile(in_path):
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.output_name:
        out_name = args.output_name
    else:
        base = os.path.basename(in_path)
        root, ext = os.path.splitext(base)
        tag = f"shuffled_{args.distribution}_r{args.corruption_rate}"
        out_name = f"{root}_{tag}{ext}"
    out_path = os.path.join(out_dir, out_name)

    n_lines = 0
    n_tokens = 0
    n_moved = 0
    n_skipped_short = 0
    t0 = time.time()
    log_every = max(1, args.log_every)

    print(f"Tokenize mode: {tok_mode}")
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            stripped = line.rstrip("\n")
            tokens = tokenize(stripped, tok_mode)
            if len(tokens) >= 2 and args.corruption_rate > 0:
                n_moved += int(round(args.corruption_rate * len(tokens)))
                tokens = shuffle_tokens(tokens, args.corruption_rate, sampler, rng)
            else:
                if len(tokens) < 2:
                    n_skipped_short += 1
            n_tokens += len(tokens)
            fout.write(detokenize(tokens, tok_mode))
            fout.write("\n")
            n_lines += 1
            if n_lines % log_every == 0:
                dt = time.time() - t0
                rate = n_lines / dt if dt > 0 else 0.0
                print(f"  processed {n_lines:>9,d} lines  ({rate:,.0f} lines/s)",
                      flush=True)

    dt = time.time() - t0
    print(f"\nDone. {n_lines:,} lines, {n_tokens:,} tokens, "
          f"~{n_moved:,} displacements in {dt:.1f}s "
          f"({n_skipped_short:,} lines with <2 tokens left unchanged)")
    print(f"Output: {out_path}")

    return {
        "input": in_path,
        "output": out_path,
        "lines": n_lines,
        "tokens": n_tokens,
        "approx_displacements": n_moved,
        "skipped_short_lines": n_skipped_short,
        "elapsed_seconds": round(dt, 2),
        "config": {
            "corruption_rate": args.corruption_rate,
            "distribution": args.distribution,
            "tokenize": tok_mode,
            "max_distance": args.max_distance,
            "geom_p": args.geom_p,
            "lam": args.lam,
            "sigma": args.sigma,
            "zipf_a": args.zipf_a,
            "seed": args.seed,
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Shuffle word order in a plain-text pretraining corpus by "
                    "displacing a random subset of tokens per line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Path to plain-text pretraining file (one document/paragraph per line).")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write the shuffled file into.")
    p.add_argument("--output_name", default=None,
                   help="Override output filename (default: derived from input + config).")

    p.add_argument("--corruption_rate", type=float, default=0.3,
                   help="Fraction of tokens per line to displace. 0 disables shuffling.")
    p.add_argument("--tokenize", choices=["whitespace", "chars"], default="whitespace",
                   help="Tokenization: 'whitespace' for EN/TR; 'chars' for unsegmented "
                        "Chinese (NLLB ZH has ~1–2 whitespace tokens/line on average).")

    p.add_argument("--distribution",
                   choices=["uniform", "geometric", "poisson", "normal", "zipf"],
                   default="geometric",
                   help="Distance distribution. Geometric is recommended (see file docstring).")
    p.add_argument("--max_distance", type=int, default=20,
                   help="Hard cap on |displacement| (in tokens).")

    p.add_argument("--geom_p", type=float, default=0.5,
                   help="Geometric distribution success probability (only used if --distribution=geometric).")
    p.add_argument("--lam", type=float, default=3.0,
                   help="Poisson rate (only used if --distribution=poisson).")
    p.add_argument("--sigma", type=float, default=3.0,
                   help="Stddev of the underlying Normal (only used if --distribution=normal).")
    p.add_argument("--zipf_a", type=float, default=2.0,
                   help="Zipf exponent (>1; only used if --distribution=zipf).")

    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility.")
    p.add_argument("--log_every", type=int, default=50000,
                   help="Print a progress line every N input lines.")
    p.add_argument("--save_config", action="store_true",
                   help="Write a <output>.config.json file alongside the shuffled output.")

    return p.parse_args()


def main():
    args = parse_args()
    summary = process_file(args)

    if args.save_config:
        cfg_path = summary["output"] + ".config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Config: {cfg_path}")


if __name__ == "__main__":
    main()
