"""Recompute the key numbers of the paper straight from the saved test predictions.

Independent of aggregate_results.py: it only reads results/davidson/<model>/seed<N>/preds_test.csv
(and, optionally, generated/errors_coded.csv) and prints
  1. Table 3   main results (mean +- sd over seeds)
  2. Table 4   paired bootstrap of BERT-base vs every other model (macro-F1)
  3. Table 6/7 ablations incl. keep-social (means, bootstrap CI, Welch p)
  4. Figure 4  mean confusion matrices (row %, rounded mean counts)
  5. Majority-class reference (accuracy, macro-F1)
  6. Table 9   counts of the manual error coding

Usage:  python verify_results.py [--results results] [--coded generated/errors_coded.csv] [--boot 2000]
Needs only numpy, pandas, scipy.
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--coded", default="generated/errors_coded.csv")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()

MODELS = [("tfidf_lr", "TF-IDF + LR"), ("lstm", "BiLSTM"), ("distilbert", "DistilBERT"), ("roberta", "RoBERTa-base"),
          ("hatebert", "HateBERT"), ("toxicbert", "toxic-bert"), ("bert", "BERT-base")]
ABL = [("bert_unweighted", "BERT, unweighted loss"), ("bert_focal", "BERT, focal loss"),
       ("bert_keepsocial", "BERT, hashtags/emoji kept")]
K = 3


def load(m):
    fs = sorted(glob.glob(os.path.join(a.results, "davidson", m, "seed*", "preds_test.csv")))
    return [pd.read_csv(f) for f in fs]


def cm(y, p):
    return np.bincount(y * K + p, minlength=K * K).reshape(K, K)


def prf(c):
    tp = np.diag(c).astype(float)
    P = tp / np.maximum(c.sum(0), 1); R = tp / np.maximum(c.sum(1), 1)
    F = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0.0)
    return P, R, F


def metrics(c):
    P, R, F = prf(c); n = c.sum(); w = c.sum(1) / n
    return dict(macroF=F.mean(), hateF=F[0], hateP=P[0], hateR=R[0], macroP=P.mean(), macroR=R.mean(),
                wF=(w * F).sum(), acc=np.trace(c) / n)


runs = {m: load(m) for m, _ in MODELS + ABL}
ref = runs["bert"]
y = ref[0]["y"].values.astype(int); n = len(y)
for m, r in runs.items():
    for d in r:
        assert (d["y"].values == y).all(), f"{m}: test rows differ from BERT-base"
print(f"Test set: {n} posts, class counts (Hate, Offensive, Clean) = {np.bincount(y).tolist()}\n")


def fmt(v):
    return f"{np.mean(v):.3f}" + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "")


print("== Table 3: main results (mean ± sd over seeds) ==")
keys = ["macroF", "hateF", "hateP", "hateR", "macroP", "macroR", "wF", "acc"]
print(f"{'Model':26s}" + "".join(f"{k:>15s}" for k in keys) + "   seeds")
for m, name in MODELS + ABL:
    if not runs[m]: continue
    M = [metrics(cm(y, d["pred"].values.astype(int))) for d in runs[m]]
    print(f"{name:26s}" + "".join(f"{fmt([x[k] for x in M]):>15s}" for k in keys) + f"   {len(M)}")

rng = np.random.default_rng(0)
idx = [rng.integers(0, n, n) for _ in range(a.boot)]


def mean_macro(rs, i):
    return np.mean([prf(cm(y[i], d["pred"].values.astype(int)[i]))[2].mean() for d in rs])


def per_seed_macro(rs):
    return np.array([prf(cm(y, d["pred"].values.astype(int)))[2].mean() for d in rs])


def compare(title, rows):
    print(f"\n== {title} ==")
    print(f"{'Comparison (variant/other minus reference or BERT-base minus other)':66s}{'diff':>8s}   95% bootstrap CI      Welch p")
    for label, sign_ref_first, other in rows:
        rs_o = runs[other]
        d = np.array([(mean_macro(ref, i) - mean_macro(rs_o, i)) if sign_ref_first else (mean_macro(rs_o, i) - mean_macro(ref, i))
                      for i in idx])
        lo, hi = np.percentile(d, [2.5, 97.5])
        s_ref, s_o = per_seed_macro(ref), per_seed_macro(rs_o)
        p = stats.ttest_ind(s_ref, s_o, equal_var=False).pvalue if len(s_o) > 1 else float("nan")
        est = (s_ref.mean() - s_o.mean()) if sign_ref_first else (s_o.mean() - s_ref.mean())
        print(f"{label:66s}{est:+8.3f}   [{lo:+.3f}, {hi:+.3f}]   {'-' if np.isnan(p) else f'{p:.3f}'}")


compare("Table 4: BERT-base minus other model, macro-F1",
        [(f"BERT-base vs {name}", True, m) for m, name in MODELS if m != "bert"])
compare("Table 7: ablation variant minus BERT-base, macro-F1",
        [(f"{name} vs BERT-base", False, m) for m, name in ABL])

print("\n== Figure 4: mean confusion matrices over seeds (row %, rounded mean counts); rows = true class ==")
for m, name in MODELS:
    if not runs[m]: continue
    c = np.mean([cm(y, d["pred"].values.astype(int)) for d in runs[m]], axis=0)
    print(name)
    for i, cl in enumerate(["Hate", "Offensive", "Clean"]):
        print(f"  {cl:10s}" + "   ".join(f"{100 * c[i, j] / c[i].sum():5.1f}% (n={c[i, j]:.0f})" for j in range(K)))

print("\n== Majority-class reference ==")
maj = cm(y, np.ones(n, dtype=int))
print(f"always 'Offensive Language': accuracy {np.trace(maj) / n:.3f}, macro-F1 {prf(maj)[2].mean():.3f}, Hate F1 {prf(maj)[2][0]:.3f}")

if os.path.exists(a.coded):
    d = pd.read_csv(a.coded)
    print("\n== Table 9: manual coding counts (from " + a.coded + ") ==")
    print(pd.crosstab(d["code"], d["type"], margins=True).to_string())
else:
    print(f"\n(no {a.coded}: Table 9 counts skipped)")
