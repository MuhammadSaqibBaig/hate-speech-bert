#!/usr/bin/env python
"""Train and evaluate every model reported in the paper.

Example (full run, needs a GPU and internet access to the Hugging Face hub):
    python run_experiments.py --out results --seeds 13 21 42 87 101 \
        --models tfidf_lr lstm bert distilbert roberta hatebert toxicbert bert_unweighted bert_focal bert_keepsocial

Cross-dataset / cross-lingual runs (train on another dataset, or XLM-R for Portuguese):
    python run_experiments.py --out results --train-on hatexplain --models bert hatebert --seeds 13 21 42
    python run_experiments.py --out results --models xlmr --seeds 13 21 42
    python run_experiments.py --out results --train-on hatebr --models xlmr --seeds 13 21 42

Finished runs are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import copy
import json
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd

import hs_common as C

# ------------------------------------------------------------------ configuration (reported in the paper)
HP = dict(max_len=128, batch_size=32, lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, max_epochs=5,
          patience=2, grad_clip=1.0, focal_gamma=2.0,
          lstm=dict(emb=128, hidden=128, dropout=0.3, max_len=64, epochs=8, batch_size=64, lr=1e-3, min_freq=2),
          tfidf=dict(ngram=(1, 2), min_df=2, C=1.0))
DEFAULT_SEEDS = [13, 21, 42, 87, 101]

MODELS = {
    "tfidf_lr": dict(kind="tfidf"),
    "lstm": dict(kind="lstm"),
    "bert": dict(kind="hf", hf="bert-base-uncased", loss="weighted", arch="bert"),
    "distilbert": dict(kind="hf", hf="distilbert-base-uncased", loss="weighted", arch="distilbert"),
    "roberta": dict(kind="hf", hf="roberta-base", loss="weighted", arch="roberta"),
    "hatebert": dict(kind="hf", hf="GroNLP/hateBERT", loss="weighted", arch="bert"),
    "toxicbert": dict(kind="hf", hf="unitary/toxic-bert", loss="weighted", arch="bert"),
    "bert_unweighted": dict(kind="hf", hf="bert-base-uncased", loss="ce", arch="bert"),
    "bert_focal": dict(kind="hf", hf="bert-base-uncased", loss="focal", arch="bert"),
    "bert_keepsocial": dict(kind="hf", hf="bert-base-uncased", loss="weighted", arch="bert", keep_social=True),
    "xlmr": dict(kind="hf", hf="xlm-roberta-base", loss="weighted", arch="roberta", multilingual=True),
}


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def env_info():
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count()}
    try:
        import torch, transformers, sklearn
        info.update(torch=torch.__version__, transformers=transformers.__version__, sklearn=sklearn.__version__)
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
    except Exception as e:  # pragma: no cover
        info["error"] = repr(e)
    return info


# ------------------------------------------------------------------ transformer models
def make_tokenizer_and_model(spec, num_labels, smoke_texts=None):
    import torch  # noqa
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, BertConfig, DistilBertConfig,
                              RobertaConfig, BertForSequenceClassification, DistilBertForSequenceClassification,
                              RobertaForSequenceClassification, PreTrainedTokenizerFast)
    if smoke_texts is None:
        tok = AutoTokenizer.from_pretrained(spec["hf"])
        model = AutoModelForSequenceClassification.from_pretrained(spec["hf"], num_labels=num_labels,
                                                                   ignore_mismatched_sizes=True)
        return tok, model
    # smoke test: tiny random model + locally trained word-piece tokenizer, no downloads
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.train_from_iterator(smoke_texts, trainers.WordPieceTrainer(
        vocab_size=1500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]")
    v = tok.vocab_size
    if spec["arch"] == "distilbert":
        cfg = DistilBertConfig(vocab_size=v, dim=32, n_layers=2, n_heads=2, hidden_dim=64, num_labels=num_labels,
                               max_position_embeddings=160)
        return tok, DistilBertForSequenceClassification(cfg)
    if spec["arch"] == "roberta":
        cfg = RobertaConfig(vocab_size=v, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                            intermediate_size=64, num_labels=num_labels, max_position_embeddings=170,
                            pad_token_id=0, bos_token_id=2, eos_token_id=3, type_vocab_size=1)
        return tok, RobertaForSequenceClassification(cfg)
    cfg = BertConfig(vocab_size=v, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                     num_labels=num_labels, max_position_embeddings=160)
    return tok, BertForSequenceClassification(cfg)


def batches(texts, tok, max_len, bs, order):
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        yield idx, enc


def predict_probs(model, tok, texts, max_len, device):
    import torch
    model.eval()
    lens = np.array([len(t) for t in texts])
    order = np.argsort(lens)  # sort by length: faster, identical results
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    with torch.no_grad():
        for idx, enc in batches(texts, tok, max_len, 64, order):
            enc = {k: v.to(device) for k, v in enc.items() if k != "token_type_ids" or model.config.model_type == "bert"}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(**enc).logits
            out[idx] = torch.softmax(logits.float(), -1).cpu().numpy()
    return out


def loss_fn(kind, logits, y, weights, gamma):
    import torch
    import torch.nn.functional as F
    if kind == "ce":
        return F.cross_entropy(logits, y)
    if kind == "weighted":
        return F.cross_entropy(logits, y, weight=weights)
    logp = F.log_softmax(logits, -1)  # focal loss, alpha = class weights
    logpt = logp.gather(1, y[:, None])[:, 0]
    pt = logpt.exp()
    return -(weights[y] * (1 - pt) ** gamma * logpt).mean()


def train_hf(spec, tr, va, seed, k, device, smoke):
    import torch
    from transformers import get_linear_schedule_with_warmup
    set_seed(seed)
    tok, model = make_tokenizer_and_model(spec, k, smoke_texts=list(tr["text"]) if smoke else None)
    model.to(device)
    max_len = HP["max_len"]
    y_tr = tr["y"].values
    w = torch.tensor(C.class_weights(y_tr, k), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    steps_per_epoch = int(np.ceil(len(tr) / HP["batch_size"]))
    total = steps_per_epoch * HP["max_epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(HP["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    texts = list(tr["text"])
    best, best_state, bad, epochs_run, t0 = -1.0, None, 0, 0, time.time()
    rng = np.random.RandomState(seed)
    for ep in range(HP["max_epochs"]):
        model.train()
        order = rng.permutation(len(texts))
        for idx, enc in batches(texts, tok, max_len, HP["batch_size"], order):
            enc = {k_: v.to(device) for k_, v in enc.items() if k_ != "token_type_ids" or model.config.model_type == "bert"}
            yb = torch.tensor(y_tr[idx], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model(**enc).logits
            loss = loss_fn(spec["loss"], logits.float(), yb, w, HP["focal_gamma"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), HP["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
        epochs_run += 1
        pv = predict_probs(model, tok, list(va["text"]), max_len, device)
        f1 = C.cm_metrics(C.confusion(va["y"].values, pv.argmax(1), k))["macro"]["F1"]
        print(f"    epoch {ep + 1}: val macro-F1 {f1:.4f}", file=sys.stderr)
        if f1 > best:
            best, bad = f1, 0
            best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    n_params = int(sum(p.numel() for p in model.parameters()))
    return model, tok, dict(train_seconds=train_seconds, epochs_run=epochs_run, best_val_macro_f1=best,
                            n_params=n_params)


# ------------------------------------------------------------------ LSTM and TF-IDF baselines
def train_lstm(tr, va, seed, k, device, smoke):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from collections import Counter
    h = HP["lstm"]
    set_seed(seed)
    cnt = Counter(w for t in tr["text"] for w in t.split())
    vocab = {w: i + 2 for i, (w, c) in enumerate(cnt.most_common()) if c >= h["min_freq"]}  # 0 pad, 1 unk

    def enc(texts):
        X = np.zeros((len(texts), h["max_len"]), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [vocab.get(w, 1) for w in t.split()][:h["max_len"]]
            X[i, :len(ids)] = ids
        return torch.tensor(X)

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(len(vocab) + 2, h["emb"], padding_idx=0)
            s.lstm = nn.LSTM(h["emb"], h["hidden"], batch_first=True, bidirectional=True)
            s.drop = nn.Dropout(h["dropout"])
            s.out = nn.Linear(2 * h["hidden"], k)

        def forward(s, x):
            m = (x != 0).unsqueeze(-1).float()
            o, _ = s.lstm(s.emb(x))
            pooled = (o * m).sum(1) / m.sum(1).clamp(min=1)
            return s.out(s.drop(pooled))

    net = Net().to(device)
    Xtr, ytr = enc(list(tr["text"])), torch.tensor(tr["y"].values)
    Xva = enc(list(va["text"]))
    w = torch.tensor(C.class_weights(tr["y"].values, k), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(net.parameters(), lr=h["lr"])
    best, best_state, t0 = -1, None, time.time()
    rng = np.random.RandomState(seed)
    epochs = 2 if smoke else h["epochs"]
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(Xtr))
        for i in range(0, len(order), h["batch_size"]):
            b = order[i:i + h["batch_size"]]
            loss = F.cross_entropy(net(Xtr[b].to(device)), ytr[b].to(device), weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xva.to(device)).argmax(1).cpu().numpy()
        f1 = C.cm_metrics(C.confusion(va["y"].values, pv, k))["macro"]["F1"]
        if f1 > best:
            best, best_state = f1, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    info = dict(train_seconds=time.time() - t0, epochs_run=epochs, best_val_macro_f1=best,
                n_params=int(sum(p.numel() for p in net.parameters())))

    def predict(texts):
        net.eval()
        with torch.no_grad():
            return F.softmax(net(enc(texts).to(device)), -1).cpu().numpy()
    return predict, info


def train_tfidf(tr, va, seed, k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    h = HP["tfidf"]
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=h["ngram"], min_df=h["min_df"], sublinear_tf=True)
    X = vec.fit_transform(tr["text"])
    clf = LogisticRegression(C=h["C"], class_weight="balanced", max_iter=2000).fit(X, tr["y"])
    info = dict(train_seconds=time.time() - t0, epochs_run=1, best_val_macro_f1=None,
                n_params=int(clf.coef_.size + clf.intercept_.size))
    return (lambda texts: clf.predict_proba(vec.transform(texts))), info


# ------------------------------------------------------------------ evaluation helpers
def timed(fn, texts, device):
    t0 = time.time()
    p = fn(texts)
    return p, (time.time() - t0) / max(len(texts), 1) * 1000.0  # ms per post


def eval_cross(name, df_eval, predict, k_model):
    """Evaluate a trained model on another dataset's test set."""
    p = predict(list(df_eval["text"]))
    pred = p.argmax(1)
    if name == "hatebr":  # binary target: collapse Hate+Offensive -> offensive
        pred = (pred != 2).astype(int) if k_model == 3 else pred
        cm = C.confusion(df_eval["y"].values, pred, 2)
    else:
        cm = C.confusion(df_eval["y"].values, pred, 3)
    return {"cm": cm.tolist(), **C.cm_metrics(cm)}


def get_data(train_on, keep_social, cache, smoke):
    """Returns dict name -> (df, split-dict, info)."""
    data = {}
    df, info = C.load_davidson(cache, keep_social)
    tr, va, te = C.split_indices(df["y"].values)
    data["davidson"] = (df, {"train": tr, "val": va, "test": te}, info)
    if not keep_social:
        hx, hinfo = C.load_hatexplain(cache)
        data["hatexplain"] = (hx, {s: np.where(hx["split"].values == s)[0] for s in ("train", "val", "test")}, hinfo)
        hb, binfo = C.load_hatebr(cache)
        btr, bva, bte = C.split_indices(hb["y"].values)
        data["hatebr"] = (hb, {"train": btr, "val": bva, "test": bte}, binfo)
    if smoke:  # tiny subsets so that the whole pipeline runs in seconds on a CPU
        for name, (d, sp, inf) in list(data.items()):
            rs = np.random.RandomState(0)
            data[name] = (d, {s: np.sort(rs.choice(ix, size=min(len(ix), 600 if s == "train" else 200), replace=False))
                              for s, ix in sp.items()}, inf)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="data_cache")
    ap.add_argument("--models", nargs="+", default=["tfidf_lr", "lstm", "bert", "distilbert", "roberta", "hatebert",
                                                     "toxicbert", "bert_unweighted", "bert_focal", "bert_keepsocial"])
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--train-on", default="davidson", choices=["davidson", "hatexplain", "hatebr"])
    ap.add_argument("--smoke", action="store_true", help="tiny random models, no downloads of weights (pipeline test)")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = env_info()
    os.makedirs(args.out, exist_ok=True)
    json.dump(env, open(os.path.join(args.out, "env.json"), "w"), indent=1)
    print("environment:", env, file=sys.stderr)

    cache_by_variant = {}
    for mname in args.models:
        spec = MODELS[mname]
        keep = spec.get("keep_social", False)
        if keep not in cache_by_variant:
            cache_by_variant[keep] = get_data(args.train_on, keep, args.cache, args.smoke)
        data = cache_by_variant[keep]
        if args.train_on != "davidson" and keep:
            continue
        df, sp, info = data[args.train_on]
        k = 2 if args.train_on == "hatebr" else 3
        tr, va, te = (df.iloc[sp[s]] for s in ("train", "val", "test"))
        if mname == "xlmr" or args.train_on == "hatebr":
            pass
        # English-only models cannot read Portuguese; only run them on English data
        if args.train_on == "hatebr" and not spec.get("multilingual"):
            continue
        seeds = [args.seeds[0]] if spec["kind"] == "tfidf" else args.seeds
        for seed in seeds:
            rdir = os.path.join(args.out, args.train_on, mname, f"seed{seed}")
            if os.path.exists(os.path.join(rdir, "metrics.json")):
                print(f"skip {rdir}", file=sys.stderr)
                continue
            print(f"== {args.train_on} | {mname} | seed {seed}", file=sys.stderr)
            if spec["kind"] == "hf":
                model, tok, tinfo = train_hf(spec, tr, va, seed, k, device, args.smoke)
                predict = lambda texts, m=model, t=tok: predict_probs(m, t, texts, HP["max_len"], device)
            elif spec["kind"] == "lstm":
                predict, tinfo = train_lstm(tr, va, seed, k, device, args.smoke)
            else:
                predict, tinfo = train_tfidf(tr, va, seed, k)
            ptest, ms_per_post = timed(predict, list(te["text"]), device)
            cm = C.confusion(te["y"].values, ptest.argmax(1), k)
            res = {"model": mname, "seed": seed, "train_on": args.train_on, "n_test": int(len(te)),
                   "n_train": int(len(tr)), "n_val": int(len(va)), "cm": cm.tolist(), **C.cm_metrics(cm),
                   "cost": {**tinfo, "infer_ms_per_post": ms_per_post,
                            "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else None},
                   "hp": {kk: vv for kk, vv in HP.items()}, "data_info": info, "env": env, "cross": {}}
            # cross-dataset evaluation (English 3-class model -> HateXplain test; multilingual -> HateBR test)
            for other in ("davidson", "hatexplain", "hatebr"):
                if other == args.train_on or other not in data:
                    continue
                odf, osp, _ = data[other]
                if other == "hatebr" and not spec.get("multilingual"):
                    continue
                if other != "hatebr" and args.train_on == "hatebr":
                    continue
                res["cross"][other] = eval_cross(other, odf.iloc[osp["test"]], predict, k)
            os.makedirs(rdir, exist_ok=True)
            pd.DataFrame({"id": te["id"].values, "y": te["y"].values, "pred": ptest.argmax(1),
                          **{f"p{c}": ptest[:, c] for c in range(k)}}).to_csv(os.path.join(rdir, "preds_test.csv"), index=False)
            json.dump(res, open(os.path.join(rdir, "metrics.json"), "w"), indent=1)
            print(f"   test acc {res['acc']:.4f} macro-F1 {res['macro']['F1']:.4f} hate-F1 {res['F1'][0]:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
