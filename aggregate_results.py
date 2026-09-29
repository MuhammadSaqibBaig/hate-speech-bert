#!/usr/bin/env python
"""Turn results/ into everything the LaTeX manuscript includes (folder generated/).

    python aggregate_results.py --results results --out generated

Every number in the paper comes out of this script.  Cells whose runs do not exist yet are printed as '--',
so the manuscript always compiles and it is obvious what is still missing.
"""
import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

import hs_common as C
import run_experiments as R

NAMES = {"tfidf_lr": "TF-IDF + LR", "lstm": "BiLSTM", "distilbert": "DistilBERT", "roberta": "RoBERTa-base",
         "hatebert": "HateBERT", "toxicbert": "toxic-bert", "bert": "BERT-base",
         "bert_unweighted": "BERT, unweighted loss", "bert_focal": "BERT, focal loss",
         "bert_keepsocial": "BERT, hashtags/emoji kept", "xlmr": "XLM-R-base"}
MAIN = ["tfidf_lr", "lstm", "distilbert", "roberta", "hatebert", "toxicbert", "bert"]
ABL = ["bert_unweighted", "bert", "bert_focal", "bert_keepsocial"]
CLS = ["Hate Speech", "Offensive Lang.", "Clean Text"]
DASH = "--"


def key(m):
    return m.replace("_", "")


def load_runs(results, train_on="davidson"):
    runs = {}
    for f in sorted(glob.glob(os.path.join(results, train_on, "*", "seed*", "metrics.json"))):
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)
    # Safety check: every run of one dataset must have been evaluated on the same test set.  Runs from an older
    # version of the code (different row set, e.g. the first keep-social ablation) are excluded and reported.
    sizes = Counter(r["n_test"] for rs in runs.values() for r in rs)
    if len(sizes) > 1:
        ref = sizes.most_common(1)[0][0]
        for m in list(runs):
            bad = [r for r in runs[m] if r["n_test"] != ref]
            if bad:
                print(f"WARNING: {len(bad)} run(s) of '{m}' ({train_on}) were evaluated on a different test set "
                      f"({bad[0]['n_test']} vs {ref} posts) and are excluded; re-run them.", file=sys.stderr)
                runs[m] = [r for r in runs[m] if r["n_test"] == ref]
                if not runs[m]:
                    del runs[m]
    return runs


def stat(vals):
    v = np.asarray(vals, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0), len(v)


def fmt(vals, d=3, pct=False):
    if not len(vals):
        return DASH
    m, s, n = stat(vals)
    if pct:
        m, s = 100 * m, 100 * s
    return f"{m:.{d}f}" if n == 1 else f"{m:.{d}f}\\,$\\pm$\\,{s:.{d}f}"


def get(runs, m, fn):
    return [fn(r) for r in runs.get(m, [])]


def tabular(cols, rows, align, extra_mid=()):
    out = ["\\begin{tabular}{" + align + "}", "\\toprule", " & ".join(cols) + " \\\\", "\\midrule"]
    for i, row in enumerate(rows):
        if i in extra_mid:
            out.append("\\midrule")
        out.append(" & ".join(row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def write(out, name, text):
    open(os.path.join(out, name), "w", encoding="utf8").write(text)


# ------------------------------------------------------------------ tables
def table_main(runs, models):
    """Macro-F1 and the Hate Speech class come first; accuracy is last because it is dominated by Offensive Language."""
    rows = []
    for m in models:
        g = lambda f: get(runs, m, f)
        rows.append([NAMES[m], fmt(g(lambda r: r["macro"]["F1"])), fmt(g(lambda r: r["F1"][0])),
                     fmt(g(lambda r: r["P"][0])), fmt(g(lambda r: r["R"][0])),
                     fmt(g(lambda r: r["macro"]["P"])), fmt(g(lambda r: r["macro"]["R"])),
                     fmt(g(lambda r: r["weighted"]["F1"])), fmt(g(lambda r: r["acc"]))])
    return tabular(["Model", "Macro-F1", "Hate F1", "Hate P", "Hate R", "Macro-P", "Macro-R", "Weighted-F1", "Accuracy"],
                   rows, "lcccccccc")


def table_perclass(runs, models):
    rows, mids = [], []
    for m in models:
        if rows:
            mids.append(len(rows))
        for c in range(3):
            g = lambda f: get(runs, m, f)
            rows.append([NAMES[m] if c == 0 else "", CLS[c], fmt(g(lambda r: r["P"][c])), fmt(g(lambda r: r["R"][c])),
                         fmt(g(lambda r: r["F1"][c])), fmt(g(lambda r: r["spec"][c]))])
    return tabular(["Model", "Class", "Precision", "Recall", "F1", "Specificity"], rows, "llcccc", mids)


def table_ablation(runs):
    rows = []
    for m in ABL:
        g = lambda f: get(runs, m, f)
        rows.append([NAMES[m], fmt(g(lambda r: r["acc"])), fmt(g(lambda r: r["macro"]["F1"])),
                     fmt(g(lambda r: r["P"][0])), fmt(g(lambda r: r["R"][0])), fmt(g(lambda r: r["F1"][0]))])
    return tabular(["Variant", "Accuracy", "Macro-F1", "Hate P", "Hate R", "Hate F1"], rows, "lccccc")


def table_cost(runs, models):
    rows = []
    for m in models:
        rs = runs.get(m, [])
        if not rs:
            rows.append([NAMES[m]] + [DASH] * 5)
            continue
        gpu = Counter(r.get("env", {}).get("gpu", "?") for r in rs).most_common(1)[0][0]
        rows.append([NAMES[m], f"{np.mean([r['cost']['n_params'] for r in rs]) / 1e6:.2f}",
                     f"{np.mean([r['cost']['epochs_run'] for r in rs]):.1f}",
                     f"{np.mean([r['cost']['train_seconds'] for r in rs]) / 60:.1f}",
                     f"{np.mean([r['cost']['infer_ms_per_post'] for r in rs]):.2f}", gpu.replace("_", "\\_")])
    return tabular(["Model", "Parameters (M)", "Epochs", "Training (min)", "Inference (ms/post)", "Device"], rows, "lccccl")


def table_config():
    hp = R.HP
    rows = [["Split", "stratified 70\\,/\\,10\\,/\\,20\\,\\% (train\\,/\\,validation\\,/\\,test), split seed " + str(C.SPLIT_SEED)
             + ", identical for all models"],
            ["Training seeds", ", ".join(map(str, R.DEFAULT_SEEDS)) + " (five runs per model)"],
            ["Maximum sequence length", str(hp["max_len"]) + " tokens"],
            ["Optimizer", "AdamW, learning rate $2\\times10^{-5}$, weight decay " + str(hp["weight_decay"])],
            ["Batch size", str(hp["batch_size"])],
            ["Schedule", "linear decay, warm-up over " + str(int(100 * hp["warmup_ratio"])) + "\\,\\% of the steps"],
            ["Epochs / early stopping", f"at most {hp['max_epochs']} epochs; best epoch by validation macro-F1; stop after {hp['patience']} epochs without improvement"],
            ["Regularization", "default dropout of each pre-trained model (0.1); gradient clipping at " + str(hp["grad_clip"])],
            ["Class weights", "$w_c = n / (K\\,n_c)$ from the training set ($K=3$); used in the weighted cross-entropy loss"],
            ["Focal loss variant", "$\\gamma = " + str(hp["focal_gamma"]) + "$, $\\alpha_c = w_c$"],
            ["BiLSTM baseline", f"embedding {hp['lstm']['emb']}, hidden {hp['lstm']['hidden']} (bidirectional), dropout {hp['lstm']['dropout']}, "
                                f"{hp['lstm']['epochs']} epochs, Adam ({hp['lstm']['lr']}), trained from scratch"],
            ["TF-IDF baseline", "word 1--2-grams, minimum document frequency 2, sublinear TF, logistic regression ($C=1$, balanced class weights)"],
            ["Precision", "mixed precision (fp16) on GPU"]]
    out = ["\\begin{tabular}{@{}lp{0.68\\linewidth}@{}}", "\\toprule", "Setting & Value \\\\", "\\midrule"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def table_cross(cache_results, results):
    dav, hx = load_runs(results, "davidson"), load_runs(results, "hatexplain")
    rows = []
    for m in ["tfidf_lr", "lstm", "distilbert", "roberta", "hatebert", "toxicbert", "bert"]:
        rd = dav.get(m, [])
        rows.append([NAMES[m] + " (train: Davidson)", fmt([r["macro"]["F1"] for r in rd]),
                     fmt([r["cross"]["hatexplain"]["macro"]["F1"] for r in rd if "hatexplain" in r["cross"]]),
                     fmt([r["cross"]["hatexplain"]["F1"][0] for r in rd if "hatexplain" in r["cross"]])])
    for m in ["bert", "hatebert"]:
        rh = hx.get(m, [])
        rows.append([NAMES[m] + " (train: HateXplain)", fmt([r["cross"]["davidson"]["macro"]["F1"] for r in rh if "davidson" in r["cross"]]),
                     fmt([r["macro"]["F1"] for r in rh]), fmt([r["F1"][0] for r in rh])])
    return tabular(["Model and training data", "Davidson test macro-F1", "HateXplain test macro-F1", "HateXplain test Hate F1"],
                   rows, "lccc", (7,))


def table_multilingual(results):
    dav, br = load_runs(results, "davidson"), load_runs(results, "hatebr")
    rows = []
    rd = dav.get("xlmr", [])
    z = [r["cross"]["hatebr"] for r in rd if "hatebr" in r["cross"]]
    rows.append(["XLM-R, trained on Davidson (English), zero-shot", fmt([c["acc"] for c in z]), fmt([c["F1"][1] for c in z]),
                 fmt([c["macro"]["F1"] for c in z])])
    rb = br.get("xlmr", [])
    rows.append(["XLM-R, trained on HateBR (Portuguese)", fmt([r["acc"] for r in rb]), fmt([r["F1"][1] for r in rb]),
                 fmt([r["macro"]["F1"] for r in rb])])
    return tabular(["Model", "Accuracy", "Offensive F1", "Macro-F1"], rows, "lccc")


# ------------------------------------------------------------------ statistics
def macro_f1_from(y, p, idx, k=3):
    cm = np.bincount(y[idx] * k + p[idx], minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(cm)
    P = np.where(cm.sum(0) > 0, tp / np.maximum(cm.sum(0), 1), 0)
    R = np.where(cm.sum(1) > 0, tp / np.maximum(cm.sum(1), 1), 0)
    F = np.where(P + R > 0, 2 * P * R / np.maximum(P + R, 1e-12), 0)
    return F.mean()


def load_preds(results, m, train_on="davidson"):
    fs = sorted(glob.glob(os.path.join(results, train_on, m, "seed*", "preds_test.csv")))
    if not fs:
        return None, None, None
    dfs = [pd.read_csv(f) for f in fs]
    return dfs[0]["id"].values, dfs[0]["y"].values, np.stack([d["pred"].values for d in dfs])


def table_stats(runs, results, ref="bert", B=2000, seed=0):
    from scipy import stats
    rows = []
    rid, ry, rp = load_preds(results, ref)
    for m in ["tfidf_lr", "lstm", "distilbert", "roberta", "hatebert", "toxicbert"]:
        if rp is None:
            rows.append([f"{NAMES[ref]} vs {NAMES[m]}", DASH, DASH, DASH])
            continue
        mid, my, mp = load_preds(results, m)
        if mp is None or not np.array_equal(mid, rid):
            rows.append([f"{NAMES[ref]} vs {NAMES[m]}", DASH, DASH, DASH])
            continue
        rng = np.random.RandomState(seed)
        n = len(ry)
        diffs = np.empty(B)
        for b in range(B):
            idx = rng.randint(0, n, n)
            diffs[b] = (np.mean([macro_f1_from(ry, p, idx) for p in rp]) - np.mean([macro_f1_from(ry, p, idx) for p in mp]))
        d0 = np.mean([macro_f1_from(ry, p, np.arange(n)) for p in rp]) - np.mean([macro_f1_from(ry, p, np.arange(n)) for p in mp])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        a = [r["macro"]["F1"] for r in runs.get(ref, [])]
        b_ = [r["macro"]["F1"] for r in runs.get(m, [])]
        pval = DASH
        if len(a) > 1 and len(b_) > 1:
            pval = f"{stats.ttest_ind(a, b_, equal_var=False).pvalue:.3f}"
            if pval == "0.000":
                pval = "$<$0.001"
        rows.append([f"{NAMES[ref]} vs {NAMES[m]}", f"{d0:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", pval])
    return tabular(["Comparison (macro-F1)", "Difference", "95\\% bootstrap CI", "Welch $p$ (seeds)"], rows, "lccc")


def table_stats_ablation(runs, results, ref="bert", B=2000, seed=0):
    """Variant minus BERT-base (positive: the variant is higher), same test posts, paired bootstrap over posts."""
    from scipy import stats
    rows, macros = [], {}
    rid, ry, rp = load_preds(results, ref)
    for m in ["bert_unweighted", "bert_focal", "bert_keepsocial"]:
        mid, my, mp = load_preds(results, m) if rp is not None else (None, None, None)
        if mp is None or not np.array_equal(mid, rid) or not np.array_equal(my, ry):
            rows.append([f"{NAMES[m]} vs {NAMES[ref]}", DASH, DASH, DASH])
            for suf in ("Diff", "Lo", "Hi", "P"):
                macros[f"abl{key(m)}{suf}"] = DASH
            continue
        rng = np.random.RandomState(seed)
        n = len(ry)
        diffs = np.empty(B)
        for b in range(B):
            idx = rng.randint(0, n, n)
            diffs[b] = np.mean([macro_f1_from(ry, p, idx) for p in mp]) - np.mean([macro_f1_from(ry, p, idx) for p in rp])
        d0 = np.mean([macro_f1_from(ry, p, np.arange(n)) for p in mp]) - np.mean([macro_f1_from(ry, p, np.arange(n)) for p in rp])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        a = [r["macro"]["F1"] for r in runs.get(m, [])]
        b_ = [r["macro"]["F1"] for r in runs.get(ref, [])]
        pval = DASH
        if len(a) > 1 and len(b_) > 1:
            pval = f"{stats.ttest_ind(a, b_, equal_var=False).pvalue:.3f}"
            if pval == "0.000":
                pval = "$<$0.001"
        rows.append([f"{NAMES[m]} vs {NAMES[ref]}", f"{d0:+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", pval])
        macros[f"abl{key(m)}Diff"] = f"{d0:+.3f}"
        macros[f"abl{key(m)}Lo"] = f"{lo:+.3f}"
        macros[f"abl{key(m)}Hi"] = f"{hi:+.3f}"
        macros[f"abl{key(m)}P"] = pval
    return tabular(["Comparison (macro-F1)", "Difference", "95\\% bootstrap CI", "Welch $p$ (seeds)"], rows, "lccc"), macros


def cross_macros(runs, results):
    """Numbers quoted in the text about external tests."""
    m = {}
    tfm = [np.mean([r["macro"]["F1"] for r in runs[k]]) for k in ("distilbert", "roberta", "hatebert", "toxicbert", "bert") if runs.get(k)]
    if tfm and runs.get("tfidf_lr"):
        m["tfMacroMin"], m["tfMacroMax"] = f"{min(tfm):.3f}", f"{max(tfm):.3f}"
        m["gapTfBest"] = f"{max(tfm) - np.mean([r['macro']['F1'] for r in runs['tfidf_lr']]):.3f}"
    dm = [np.mean([r["macro"]["F1"] for r in runs[k]]) for k in MAIN if runs.get(k)]
    if dm:
        m["davMacroMin"], m["davMacroMax"] = f"{min(dm):.2f}", f"{max(dm):.2f}"
    if runs.get("bert"):
        cm = np.mean([np.array(r["cm"], float) for r in runs["bert"]], axis=0)
        m["bertHateToOff"] = f"{100 * cm[0, 1] / cm[0].sum():.0f}"
        m["bertHateToClean"] = f"{100 * cm[0, 2] / cm[0].sum():.0f}"
    hx, br = load_runs(results, "hatexplain"), load_runs(results, "hatebr")
    def mean(v): return float(np.mean(v))
    rd = runs.get("bert", [])
    z = [r["cross"]["hatexplain"]["macro"]["F1"] for r in rd if "hatexplain" in r["cross"]]
    if z: m["xBertHxMacro"] = f"{mean(z):.3f}"
    lo = [mean([r["cross"]["hatexplain"]["macro"]["F1"] for r in runs[k] if "hatexplain" in r["cross"]]) for k in MAIN if runs.get(k)]
    if lo: m["xHxMacroMin"], m["xHxMacroMax"] = f"{min(lo):.2f}", f"{max(lo):.2f}"
    h = [r["cross"]["davidson"]["macro"]["F1"] for r in hx.get("bert", []) if "davidson" in r["cross"]]
    if h: m["xHxToDavMacro"] = f"{mean(h):.3f}"
    h2 = [r["macro"]["F1"] for r in hx.get("bert", [])]
    if h2: m["xHxInMacro"] = f"{mean(h2):.3f}"
    zz = [r["cross"]["hatebr"]["macro"]["F1"] for r in runs.get("xlmr", []) if "hatebr" in r["cross"]]
    if zz: m["xBrZeroMacro"] = f"{mean(zz):.3f}"
    zi = [r["macro"]["F1"] for r in br.get("xlmr", [])]
    if zi: m["xBrInMacro"] = f"{mean(zi):.3f}"
    return m


# ------------------------------------------------------------------ error analysis
def error_analysis(results, cache, out, model="bert"):
    ids, y, P = load_preds(results, model)
    empty = tabular(["Measure", "Hate detected", "Hate missed", "False alarms"],
                    [[m_, DASH, DASH, DASH] for m_ in ["Posts", "Annotator agreement", "Unanimous annotators (\\%)",
                                                        "Vote share for Hate (\\%)", "Length (tokens)",
                                                        "Offensive-leaning vocabulary (\\%)", "Missed by all seeds (\\%)"]], "lccc")
    if P is None or not os.path.exists(os.path.join(cache, "davidson_labeled_data.csv")):
        return empty, {}
    from scipy import stats
    df, _ = C.load_davidson(cache)
    df = df.set_index("id").loc[ids].reset_index()
    tr_idx, _, _ = C.split_indices(C.load_davidson(cache)[0]["y"].values)
    full = C.load_davidson(cache)[0]
    train = full.iloc[tr_idx]
    # offensive-leaning vocabulary: highest smoothed log-odds Offensive vs Hate in the training set
    ch = Counter(w for t in train[train.y == 0]["text"] for w in t.split())
    co = Counter(w for t in train[train.y == 1]["text"] for w in t.split())
    nh, no = sum(ch.values()), sum(co.values())
    vocab = {w for w in set(ch) | set(co) if ch[w] + co[w] >= 20}
    lo = {w: np.log((co[w] + 0.5) / (no + 0.5 * len(vocab))) - np.log((ch[w] + 0.5) / (nh + 0.5 * len(vocab))) for w in vocab}
    offvocab = set(sorted(lo, key=lambda w: (-lo[w], w))[:150])  # ties broken alphabetically: deterministic
    frac_not_hate = (P != 0).mean(0)
    hate, off = y == 0, y == 1
    missed = hate & (frac_not_hate >= 0.5)
    found = hate & (frac_not_hate < 0.5)
    alarm = off & ((P == 0).mean(0) >= 0.5)
    df["agree"] = df[["v_hate", "v_off", "v_clean"]].max(axis=1) / df["n_votes"]
    df["unan"] = (df[["v_hate", "v_off", "v_clean"]].max(axis=1) == df["n_votes"]).astype(float)
    df["hshare"] = df["v_hate"] / df["n_votes"]
    df["len"] = df["text"].str.split().str.len()
    df["offshare"] = [np.mean([w in offvocab for w in t.split()]) if t.split() else 0 for t in df["text"]]
    groups = {"found": found, "missed": missed, "alarm": alarm}

    def col(name, f, d=2, pct=False):
        cells = []
        for g in ("found", "missed", "alarm"):
            v = f(df[groups[g]])
            cells.append(DASH if np.isnan(v) else (f"{100 * v:.1f}" if pct else f"{v:.{d}f}"))
        return [name] + cells
    rows = [["Posts"] + [str(int(groups[g].sum())) for g in ("found", "missed", "alarm")],
            col("Annotator agreement", lambda d: d["agree"].mean()),
            col("Unanimous annotators (\\%)", lambda d: d["unan"].mean(), pct=True),
            col("Vote share for Hate (\\%)", lambda d: d["hshare"].mean(), pct=True),
            col("Length (tokens)", lambda d: d["len"].mean(), 1),
            col("Offensive-leaning vocabulary (\\%)", lambda d: d["offshare"].mean(), pct=True)]
    all_missed = float(np.mean((P[:, missed] != 0).all(0)) * 100) if missed.sum() else 0.0
    rows.append(["Missed by all seeds (\\%)", DASH, f"{all_missed:.1f}", DASH])
    macros = {}
    if found.sum() > 1 and missed.sum() > 1:
        macros["errAgreeP"] = f"{stats.mannwhitneyu(df[found]['agree'], df[missed]['agree']).pvalue:.3g}"
        macros["errOffP"] = f"{stats.mannwhitneyu(df[found]['offshare'], df[missed]['offshare']).pvalue:.3g}"
    macros["errAgreeFound"] = f"{df[found]['agree'].mean():.2f}"
    macros["errAgreeMissed"] = f"{df[missed]['agree'].mean():.2f}"
    macros["errUnanFound"] = f"{100 * df[found]['unan'].mean():.0f}"
    macros["errUnanMissed"] = f"{100 * df[missed]['unan'].mean():.0f}"
    macros["errOffFound"] = f"{100 * df[found]['offshare'].mean():.1f}"
    macros["errOffMissed"] = f"{100 * df[missed]['offshare'].mean():.1f}"
    macros["errAllSeeds"] = f"{all_missed:.0f}"
    macros["errAlarm"] = str(int(alarm.sum()))
    macros["errMissed"] = str(int(missed.sum()))
    macros["errHate"] = str(int(hate.sum()))
    # manual-coding sheet: 40 missed Hate posts + 20 false alarms, fixed random sample
    rs = np.random.RandomState(7)
    pick = list(rs.choice(np.where(missed)[0], size=min(40, int(missed.sum())), replace=False)) + \
        list(rs.choice(np.where(alarm)[0], size=min(20, int(alarm.sum())), replace=False))
    sheet = df.iloc[pick][["id", "raw", "text", "v_hate", "v_off", "v_clean"]].copy()
    sheet.insert(1, "type", ["missed_hate"] * min(40, int(missed.sum())) + ["false_alarm"] * min(20, int(alarm.sum())))
    sheet["code"] = ""
    sheet["notes"] = ""
    sheet.to_csv(os.path.join(out, "errors_to_code.csv"), index=False)
    return tabular(["Measure", "Hate detected", "Hate missed", "False alarms"], rows, "lccc"), macros


CODES = ["implicit_hate", "sarcasm_irony", "quoted_or_reported", "reclaimed_ingroup", "label_ambiguous",
         "explicit_but_untargeted", "other"]
CODE_NAMES = {"implicit_hate": "Implicit hate (no explicit slur)", "sarcasm_irony": "Sarcasm or irony",
              "quoted_or_reported": "Quoted or reported speech", "reclaimed_ingroup": "Reclaimed / in-group use",
              "label_ambiguous": "Label debatable", "explicit_but_untargeted": "Explicit insult, no protected target",
              "other": "Other"}


def table_error_codes(out):
    p = os.path.join(out, "errors_coded.csv")
    rows = []
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[d["code"].isin(CODES)]
        for t in ("missed_hate", "false_alarm"):
            sub = d[d["type"] == t]
            for c in CODES:
                rows.append([("Missed Hate" if t == "missed_hate" else "False alarms") if c == CODES[0] else "",
                             CODE_NAMES[c], str(int((sub["code"] == c).sum())),
                             f"{100 * (sub['code'] == c).mean():.0f}" if len(sub) else DASH])
    else:
        rows = [["Missed Hate" if i == 0 else "", CODE_NAMES[c], DASH, DASH] for i, c in enumerate(CODES)]
    return tabular(["Error type", "Category", "Posts", "\\%"], rows, "llcc", (len(CODES),))


# ------------------------------------------------------------------ figures
def figures(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def draw(ax, m):
        rs = runs.get(m, [])
        if not rs:
            ax.text(0.5, 0.5, "not run yet", ha="center", va="center"); ax.axis("off"); return
        cm = np.mean([np.array(r["cm"], float) for r in rs], axis=0)
        pr = cm / cm.sum(1, keepdims=True)
        ax.imshow(pr, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * pr[i, j]:.1f}%\n(n={cm[i, j]:.0f})", ha="center", va="center", fontsize=7,
                        color="white" if pr[i, j] > 0.5 else "black")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(["Hate", "Offensive", "Clean"], fontsize=8); ax.set_yticklabels(["Hate", "Offensive", "Clean"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True label", fontsize=8)
        ax.set_title(NAMES[m], fontsize=9)
    for m in MAIN:
        fig, ax = plt.subplots(figsize=(3.3, 3.0)); draw(ax, m); fig.tight_layout()
        fig.savefig(os.path.join(out, f"figure_cm_{key(m)}.pdf")); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2))
    for ax, m in zip(axes.ravel(), MAIN):
        draw(ax, m)
    for ax in axes.ravel()[len(MAIN):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "figure_cm_all.pdf")); plt.close(fig)


def figure_class_dist(cache, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = os.path.join(cache, "davidson_labeled_data.csv")
    if not os.path.exists(p):
        return
    df, _ = C.load_davidson(cache)
    cnt = df["y"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    bars = ax.bar(["Hate\nSpeech", "Offensive\nLanguage", "Clean\nText"], cnt.values, color=["#d62728", "#f2c744", "#2ca02c"])
    for b, v in zip(bars, cnt.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 250, f"{v:,}\n({100 * v / cnt.sum():.1f}%)", ha="center", fontsize=8)
    ax.set_ylabel("Posts", fontsize=9); ax.set_ylim(0, cnt.max() * 1.18)
    ax.tick_params(axis="x", labelsize=10); ax.tick_params(axis="y", labelsize=9)  # horizontal, readable labels
    fig.tight_layout(); fig.savefig(os.path.join(out, "figure_class_dist.pdf")); plt.close(fig)


# ------------------------------------------------------------------ macros
def numbers(runs, cache, results, out, extra):
    lines = ["\\providecommand{\\res}[2]{\\csname res@#1@#2\\endcsname}",
             "\\providecommand{\\setres}[3]{\\expandafter\\gdef\\csname res@#1@#2\\endcsname{#3}}"]

    def setres(k, name, v):
        lines.append(f"\\setres{{{k}}}{{{name}}}{{{v}}}")
    metrics = {"Acc": (lambda r: r["acc"], True, 1), "MacroP": (lambda r: r["macro"]["P"], False, 3),
               "MacroR": (lambda r: r["macro"]["R"], False, 3), "MacroF": (lambda r: r["macro"]["F1"], False, 3),
               "WeightedF": (lambda r: r["weighted"]["F1"], False, 3), "HateP": (lambda r: r["P"][0], False, 3),
               "HateR": (lambda r: r["R"][0], False, 3), "HateF": (lambda r: r["F1"][0], False, 3),
               "OffF": (lambda r: r["F1"][1], False, 3), "CleanF": (lambda r: r["F1"][2], False, 3)}
    for m in R.MODELS:
        k = key(m)
        rs = runs.get(m, [])
        setres(k, "name", NAMES[m])
        setres(k, "n", len(rs) if rs else DASH)
        for name, (f, pct, d) in metrics.items():
            if rs:
                mean, sd, _ = stat([f(r) for r in rs])
                setres(k, name, f"{100 * mean:.{d}f}" if pct else f"{mean:.{d}f}")
                setres(k, name + "Sd", f"{100 * sd:.{d}f}" if pct else f"{sd:.{d}f}")
            else:
                setres(k, name, DASH); setres(k, name + "Sd", DASH)
        if rs:
            setres(k, "Params", f"{np.mean([r['cost']['n_params'] for r in rs]) / 1e6:.0f}")
            setres(k, "TrainMin", f"{np.mean([r['cost']['train_seconds'] for r in rs]) / 60:.1f}")
            setres(k, "InferMs", f"{np.mean([r['cost']['infer_ms_per_post'] for r in rs]):.2f}")
        else:
            for name in ("Params", "TrainMin", "InferMs"):
                setres(k, name, DASH)
    have = [m for m in MAIN if runs.get(m)]
    if have:
        b = max(have, key=lambda m: np.mean([r["macro"]["F1"] for r in runs[m]]))
        h = max(have, key=lambda m: np.mean([r["F1"][0] for r in runs[m]]))
        lines.append(f"\\gdef\\bestModel{{{NAMES[b]}}}\\gdef\\bestMacroF{{{np.mean([r['macro']['F1'] for r in runs[b]]):.3f}}}")
        lines.append(f"\\gdef\\bestHateModel{{{NAMES[h]}}}\\gdef\\bestHateF{{{np.mean([r['F1'][0] for r in runs[h]]):.3f}}}")
        seeds = sorted({r["seed"] for m in have for r in runs[m]})
        lines.append(f"\\gdef\\nSeeds{{{max(len(runs[m]) for m in have)}}}")
    else:
        lines += ["\\gdef\\bestModel{--}\\gdef\\bestMacroF{--}\\gdef\\bestHateModel{--}\\gdef\\bestHateF{--}\\gdef\\nSeeds{5}"]
    # data macros
    info = next((runs[m][0]["data_info"] for m in R.MODELS if runs.get(m)), None)
    p = os.path.join(cache, "davidson_labeled_data.csv")
    if info and os.path.exists(p):
        df, _ = C.load_davidson(cache)
        cnt = df["y"].value_counts().sort_index()
        n = len(df)
        vals = {"dataRowsFile": info["rows_in_file"], "dataEmpty": info["rows_empty_after_cleaning"], "dataN": n,
                "dataHate": int(cnt[0]), "dataOff": int(cnt[1]), "dataClean": int(cnt[2]),
                "dataHatePct": f"{100 * cnt[0] / n:.2f}", "dataOffPct": f"{100 * cnt[1] / n:.2f}",
                "dataCleanPct": f"{100 * cnt[2] / n:.2f}", "dataSha": info["file_sha256"]}
        r0 = next(runs[m][0] for m in R.MODELS if runs.get(m))
        vals.update(dataTrain=r0["n_train"], dataVal=r0["n_val"], dataTest=r0["n_test"])
    else:
        vals = {k: DASH for k in ["dataRowsFile", "dataEmpty", "dataN", "dataHate", "dataOff", "dataClean", "dataHatePct",
                                  "dataOffPct", "dataCleanPct", "dataSha", "dataTrain", "dataVal", "dataTest"]}
        vals["dataSha"] = C.DAVIDSON_SHA256
    if os.path.exists(p):
        df_, _ = C.load_davidson(cache)
        _, _, te_ = C.split_indices(df_["y"].values)
        yt = df_["y"].values[te_]
        cnt_t = np.bincount(yt, minlength=3)
        cm_maj = np.zeros((3, 3), dtype=int)
        cm_maj[:, 1] = cnt_t  # a classifier that always says "Offensive Language"
        mm = C.cm_metrics(cm_maj)
        vals.update(testHate=int(cnt_t[0]), testOff=int(cnt_t[1]), testClean=int(cnt_t[2]),
                    majAcc=f"{100 * mm['acc']:.1f}", majMacroF=f"{mm['macro']['F1']:.2f}")
    for k, v in vals.items():
        lines.append(f"\\gdef\\{k}{{{v}}}")
    for k, v in extra.items():
        lines.append(f"\\gdef\\{k}{{{v}}}")
    # hardware / software (taken from the transformer runs)
    env = next((runs["bert"][0].get("env", {}) for _ in [0] if runs.get("bert")), None) or {}
    for k, v in {"envGpu": env.get("gpu", DASH), "envTorch": env.get("torch", DASH),
                 "envTransformers": env.get("transformers", DASH), "envPython": env.get("python", DASH)}.items():
        lines.append(f"\\gdef\\{k}{{{str(v).replace('_', ' ')}}}")
    write(out, "numbers.tex", "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="generated")
    ap.add_argument("--cache", default="data_cache")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = load_runs(a.results)
    write(a.out, "table_main.tex", table_main(runs, MAIN))
    write(a.out, "table_perclass_bert.tex", table_perclass(runs, ["bert"]))
    write(a.out, "table_perclass_all.tex", table_perclass(runs, MAIN))
    write(a.out, "table_ablation.tex", table_ablation(runs))
    write(a.out, "table_cost.tex", table_cost(runs, MAIN))
    write(a.out, "table_config.tex", table_config())
    write(a.out, "table_cross.tex", table_cross(a.cache, a.results))
    write(a.out, "table_multilingual.tex", table_multilingual(a.results))
    write(a.out, "table_stats.tex", table_stats(runs, a.results))
    tab_abl, abl_macros = table_stats_ablation(runs, a.results)
    write(a.out, "table_stats_ablation.tex", tab_abl)
    tab, extra = error_analysis(a.results, a.cache, a.out)
    extra.update(abl_macros)
    extra.update(cross_macros(runs, a.results))
    write(a.out, "table_errors.tex", tab)
    write(a.out, "table_error_codes.tex", table_error_codes(a.out))
    figures(runs, a.out)
    figure_class_dist(a.cache, a.out)
    numbers(runs, a.cache, a.results, a.out, extra)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
