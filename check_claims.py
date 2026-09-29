#!/usr/bin/env python
"""Check the qualitative statements the manuscript makes against the actual results.

    python check_claims.py --results results

The prose in the Discussion and Conclusion was written so that it does not depend on which model wins.  The
statements below ARE asserted in the text, though; if one prints FAIL, edit the matching sentence.
"""
import argparse
import numpy as np
import aggregate_results as A


def mean(runs, m, f):
    return np.mean([f(r) for r in runs[m]]) if runs.get(m) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    runs = A.load_runs(a.results)
    hx = A.load_runs(a.results, "hatexplain")
    checks = []

    def add(label, ok, sentence):
        checks.append((label, ok, sentence))

    have = [m for m in A.MAIN if runs.get(m)]
    add("C1 Hate Speech has the lowest F1 of the three classes for every model",
        all(min(range(3), key=lambda c: mean(runs, m, lambda r: r["F1"][c])) == 0 for m in have) if have else None,
        "Discussion, paragraph 1")
    tf = [m for m in ("distilbert", "roberta", "hatebert", "toxicbert", "bert") if runs.get(m)]
    add("C2 every transformer beats TF-IDF+LR and BiLSTM on macro-F1",
        all(mean(runs, m, lambda r: r["macro"]["F1"]) > max(mean(runs, b, lambda r: r["macro"]["F1"]) for b in ("tfidf_lr", "lstm") if runs.get(b))
            for m in tf) if tf and runs.get("tfidf_lr") and runs.get("lstm") else None, "Results / Discussion")
    if runs.get("bert") and runs.get("bert_unweighted"):
        add("C3 class weighting raises Hate Speech recall relative to unweighted loss",
            mean(runs, "bert", lambda r: r["R"][0]) > mean(runs, "bert_unweighted", lambda r: r["R"][0]), "Ablation paragraph")
        add("C4 class weighting does not lower macro-F1 relative to unweighted loss",
            mean(runs, "bert", lambda r: r["macro"]["F1"]) >= mean(runs, "bert_unweighted", lambda r: r["macro"]["F1"]) - 0.005,
            "Ablation paragraph")
    else:
        add("C3/C4 class-weighting ablation", None, "Ablation paragraph")
    if runs.get("bert"):
        add("C5 most missed Hate posts are predicted as Offensive Language, not Clean",
            mean(runs, "bert", lambda r: r["cm"][0][1]) > mean(runs, "bert", lambda r: r["cm"][0][2]), "Error analysis")
    xs = [m for m in ("bert", "roberta", "hatebert", "distilbert", "toxicbert", "lstm", "tfidf_lr")
          if runs.get(m) and all("hatexplain" in r["cross"] for r in runs[m])]
    add("C6 macro-F1 on HateXplain (zero-shot) is below in-domain macro-F1 for every model",
        all(mean(runs, m, lambda r: r["cross"]["hatexplain"]["macro"]["F1"]) < mean(runs, m, lambda r: r["macro"]["F1"]) for m in xs) if xs else None,
        "Generalization paragraph")
    if have:
        acc = [mean(runs, m, lambda r: r["acc"]) for m in have]
        hf = [mean(runs, m, lambda r: r["F1"][0]) for m in have]
        add("C7 the spread of accuracy across models is smaller than the spread of Hate Speech F1",
            (max(acc) - min(acc)) < (max(hf) - min(hf)), "Discussion, paragraph 1")
    # C8/C9: what the text says about the paired-bootstrap intervals (recomputed with the same code as Table 3)
    import re
    tab = A.table_stats(A.load_runs(a.results), a.results)
    tab_abl, _ = A.table_stats_ablation(A.load_runs(a.results), a.results)
    def ci(table, name):
        for line in table.splitlines():
            if name in line:
                m_ = re.search(r"\[([+-][\d.]+), ([+-][\d.]+)\]", line)
                return (float(m_.group(1)), float(m_.group(2))) if m_ else None
        return None
    others = [ci(tab, "vs " + n) for n in ("DistilBERT", "RoBERTa-base", "HateBERT", "toxic-bert")]
    add("C8 the interval of BERT-base against each of the other four transformers includes zero (no ranking among transformers)",
        all(c and c[0] <= 0 <= c[1] for c in others) if all(others) else None, "Results, Discussion, Abstract, Conclusion")
    base = [ci(tab, "vs TF-IDF"), ci(tab, "vs BiLSTM")]
    add("C9 BERT-base is above TF-IDF+LR and BiLSTM (interval excludes zero)",
        all(c[0] > 0 for c in base) if all(base) else None, "Results, Abstract")
    ab = [ci(tab_abl, "unweighted"), ci(tab_abl, "focal loss")]
    add("C10 unweighted and focal-loss variants are not distinguishable from BERT-base on macro-F1",
        all(c[0] <= 0 <= c[1] for c in ab) if all(ab) else None, "Ablation paragraph")
    ks = ci(tab_abl, "hashtags/emoji")
    add("C11 keep-social: interval EXCLUDES zero (only then may the text call it an improvement/effect)",
        (ks[0] > 0 or ks[1] < 0) if ks else None, "Ablation paragraph and Discussion")
    for label, ok, where in checks:
        print(f"{'PASS' if ok else ('n/a ' if ok is None else 'FAIL')}  {label}   [{where}]")


if __name__ == "__main__":
    main()
