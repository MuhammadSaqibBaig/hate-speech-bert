"""Compute Cohen's kappa for two independent coders and prepare the agreed coding sheet.

Step 1:  python finalize_coding.py coder1.csv coder2.csv
         -> prints kappa (paste into \\errKappa in main.tex) and writes disagreements.csv
Step 2:  discuss the rows in disagreements.csv, fill the `final` column, then
         python finalize_coding.py coder1.csv coder2.csv --resolved disagreements.csv
         -> writes generated/errors_coded.csv (agreed codes). Then re-run aggregate_results.py.
"""
import argparse, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

CODES = ["implicit_hate", "sarcasm_irony", "quoted_or_reported", "reclaimed_ingroup",
         "label_ambiguous", "explicit_but_untargeted", "other"]

ap = argparse.ArgumentParser()
ap.add_argument("coder1"); ap.add_argument("coder2")
ap.add_argument("--resolved", default=None)
ap.add_argument("--out", default="generated/errors_coded.csv")
a = ap.parse_args()

c1, c2 = pd.read_csv(a.coder1), pd.read_csv(a.coder2)
m = c1[["id", "type", "code"]].merge(c2[["id", "code"]], on="id", suffixes=("_1", "_2"))
if len(m) != 60 or m["code_1"].isna().any() or m["code_2"].isna().any():
    sys.exit(f"Both files must have all 60 posts coded (found {len(m)} rows, "
             f"{int(m['code_1'].isna().sum() + m['code_2'].isna().sum())} empty codes).")
bad = (set(m["code_1"]) | set(m["code_2"])) - set(CODES)
if bad:
    sys.exit(f"Unknown codes: {sorted(bad)}. Allowed: {CODES}")

k = cohen_kappa_score(m["code_1"], m["code_2"])
agree = (m["code_1"] == m["code_2"]).mean()
print(f"Cohen's kappa (all 60 posts): {k:.2f}   raw agreement: {100*agree:.0f}%")
for t in ("missed_hate", "false_alarm"):
    s = m[m["type"] == t]
    print(f"  {t}: n={len(s)}, agreement {100*(s.code_1==s.code_2).mean():.0f}%")
print(f"-> In main.tex use:  \\newcommand{{\\errKappa}}{{{k:.2f}}}")

dis = m[m["code_1"] != m["code_2"]].copy()
if a.resolved is None:
    dis["final"] = ""
    dis.to_csv("disagreements.csv", index=False)
    print(f"{len(dis)} disagreements written to disagreements.csv; fill the `final` column after discussion.")
else:
    r = pd.read_csv(a.resolved)
    fin = m.set_index("id")
    fin["code"] = fin["code_1"]
    for _, row in r.iterrows():
        if pd.isna(row["final"]) or row["final"] not in CODES:
            sys.exit(f"Row id={row['id']} has no valid `final` code.")
        fin.loc[row["id"], "code"] = row["final"]
    fin.reset_index()[["id", "type", "code"]].to_csv(a.out, index=False)
    print(f"Agreed coding written to {a.out}. Now run: python aggregate_results.py --results results --out generated --cache data_cache")
