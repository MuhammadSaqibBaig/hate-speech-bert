"""Shared data loading, preprocessing, splitting and metric code.

Everything that turns raw files into numbers lives here, so that all models are
trained, evaluated and reported through exactly the same code path.
"""
import hashlib
import html
import json
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["Hate Speech", "Offensive Language", "Clean Text"]  # ids 0, 1, 2
SPLIT_SEED = 42  # one fixed split for every model and every training seed

DAVIDSON_URL = ("https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/"
                "master/data/labeled_data.csv")
DAVIDSON_SHA256 = "fcb8bc7c68120ae4af04a5b9acd58585513ede11e1548ebf36a5c2040b6f6281"
HATEXPLAIN_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
HATEXPLAIN_SPLIT_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"
HATEBR_URL = "https://raw.githubusercontent.com/franciellevargas/HateBR/main/dataset/HateBR.csv"

# Small normalisation dictionary for common social-media abbreviations (the full list is this one).
SLANG = {"u": "you", "ur": "your", "r": "are", "gr8": "great", "b4": "before", "thx": "thanks",
         "pls": "please", "plz": "please", "idk": "i do not know", "imo": "in my opinion",
         "omg": "oh my god", "cuz": "because", "bc": "because", "tho": "though", "ya": "you",
         "smh": "shaking my head", "tbh": "to be honest"}


def download(url, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {url}", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(t, keep_social=False):
    """Preprocessing used for all English data.

    Default: remove URLs, @mentions, #hashtags, emoji and special characters, lowercase, and
    expand the abbreviations in SLANG.  keep_social=True is the ablation that keeps hashtag
    words, replaces mentions by '@user' and URLs by 'http', and keeps emoji.
    """
    t = html.unescape(str(t))
    t = re.sub(r"^\s*rt\s+", "", t, flags=re.I)
    if keep_social:
        t = re.sub(r"https?://\S+|www\.\S+", "http", t)
        t = re.sub(r"@\w+", "@user", t)
        t = re.sub(r"#(\w+)", r"\1", t)
        t = t.lower()
        t = "".join(c if (c.isalnum() or c.isspace() or c in "@'!?.," or ord(c) >= 0x2190) else " " for c in t)
    else:
        t = re.sub(r"https?://\S+|www\.\S+", "", t)
        t = re.sub(r"@\w+", "", t)
        t = re.sub(r"#\w+", "", t)
        t = t.lower()
        t = re.sub(r"[^a-z0-9\s']", " ", t)
    return " ".join(SLANG.get(w, w) for w in t.split())


def clean_pt(t):
    t = html.unescape(str(t))
    t = re.sub(r"https?://\S+|www\.\S+", "", t)
    t = re.sub(r"@\w+", "@user", t)
    return " ".join(t.lower().split())


def load_davidson(cache_dir, keep_social=False):
    path = download(DAVIDSON_URL, os.path.join(cache_dir, "davidson_labeled_data.csv"))
    digest = sha256(path)
    if digest != DAVIDSON_SHA256:
        print(f"WARNING: Davidson file hash {digest} differs from the one used in the paper", file=sys.stderr)
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "id": raw.iloc[:, 0].values,
        "raw": raw["tweet"].values,
        "y": raw["class"].values.astype(int),  # 0 hate, 1 offensive, 2 neither (= clean)
        "n_votes": raw["count"].values,
        "v_hate": raw["hate_speech"].values,
        "v_off": raw["offensive_language"].values,
        "v_clean": raw["neither"].values,
    })
    n_all = len(df)
    # The row set (and hence the split) is defined by the default cleaning, so that every variant, including the
    # keep-social ablation, is trained and tested on exactly the same posts.
    keep_rows = np.array([len(clean_text(t, False)) > 0 for t in df["raw"]])
    df = df[keep_rows].reset_index(drop=True)
    df["text"] = [clean_text(t, keep_social) for t in df["raw"]]
    assert (df["text"].str.len() > 0).all(), "a post became empty under the keep-social cleaning"
    info = {"file_sha256": digest, "rows_in_file": int(n_all), "rows_empty_after_cleaning": int(n_all - len(df)),
            "rows_used": int(len(df))}
    return df, info


def load_hatexplain(cache_dir):
    data = json.load(open(download(HATEXPLAIN_URL, os.path.join(cache_dir, "hatexplain_dataset.json")), encoding="utf8"))
    div = json.load(open(download(HATEXPLAIN_SPLIT_URL, os.path.join(cache_dir, "hatexplain_divisions.json")), encoding="utf8"))
    lab = {"hatespeech": 0, "offensive": 1, "normal": 2}
    split_of = {pid: s for s, ids in div.items() for pid in ids}
    rows, dropped = [], 0
    for pid, rec in data.items():
        votes = [lab[a["label"]] for a in rec["annotators"]]
        counts = np.bincount(votes, minlength=3)
        if counts.max() < 2:  # three different labels: no majority, dropped as in the original paper
            dropped += 1
            continue
        rows.append({"id": pid, "y": int(counts.argmax()), "text": clean_text(" ".join(rec["post_tokens"])),
                     "split": {"val": "val"}.get(split_of.get(pid, "train"), split_of.get(pid, "train"))})
    df = pd.DataFrame(rows)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df, {"posts_without_majority_dropped": dropped, "rows_used": int(len(df))}


def load_hatebr(cache_dir):
    raw = pd.read_csv(download(HATEBR_URL, os.path.join(cache_dir, "hatebr.csv")))
    df = pd.DataFrame({"id": raw["id"].values, "y": raw["label_final"].values.astype(int),
                       "text": [clean_pt(t) for t in raw["comentario"]]})
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df, {"rows_used": int(len(df))}  # binary: 1 = offensive


def split_indices(y, seed=SPLIT_SEED):
    """Stratified 70 / 10 / 20 split into train / validation / test."""
    idx = np.arange(len(y))
    tr_val, te = train_test_split(idx, test_size=0.20, stratify=y, random_state=seed)
    tr, va = train_test_split(tr_val, test_size=0.125, stratify=np.asarray(y)[tr_val], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def class_weights(y, k):
    counts = np.bincount(y, minlength=k).astype(float)
    return counts.sum() / (k * np.maximum(counts, 1))  # n / (k * n_c)


def cm_metrics(cm):
    """All reported numbers come from a confusion matrix (rows = true, columns = predicted)."""
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = n - tp - fp - fn
    safe = lambda a, b: np.where(b > 0, a / np.maximum(b, 1e-12), 0.0)
    P, R = safe(tp, tp + fp), safe(tp, tp + fn)
    F = safe(2 * P * R, P + R)
    S = safe(tn, tn + fp)
    sup = cm.sum(1)
    w = sup / sup.sum()
    return {
        "acc": float(tp.sum() / n), "support": sup.tolist(),
        "P": P.tolist(), "R": R.tolist(), "F1": F.tolist(), "spec": S.tolist(),
        "macro": {"P": float(P.mean()), "R": float(R.mean()), "F1": float(F.mean()), "spec": float(S.mean())},
        "weighted": {"P": float((P * w).sum()), "R": float((R * w).sum()), "F1": float((F * w).sum()),
                     "spec": float((S * w).sum())},
    }


def confusion(y_true, y_pred, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm
