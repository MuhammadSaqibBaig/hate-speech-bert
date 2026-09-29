# Hate speech detection with fine-tuned BERT: experiments

This folder reproduces every number, table and figure in the paper. Nothing in the manuscript is typed in by hand:
`aggregate_results.py` writes the LaTeX tables, figures and number macros that `main.tex` includes.

## What is compared
TF-IDF + logistic regression, a BiLSTM, DistilBERT, RoBERTa-base, HateBERT, toxic-bert and BERT-base (main comparison);
BERT with unweighted loss, focal loss, and with hashtags/emoji kept (ablations); HateXplain (other platform) and HateBR
(Brazilian Portuguese, via XLM-R) for generalization. Five training seeds per model, one fixed 70/10/20 stratified split.

## Run it
On Google Colab open `colab_run_all.ipynb`, on Kaggle open `kaggle_run_all.ipynb` (GPU runtime): each contains this code and runs every step below. The folder `results/` holds the per-run metrics and test predictions behind every table of the paper (run on a Tesla T4), so the tables can be regenerated without retraining: `python aggregate_results.py --results results --out generated`.

On your own machine:
```bash
pip install -r requirements.txt            # a CUDA GPU is strongly recommended (Colab T4 is enough)
# 1. main comparison, five seeds each (about 5-8 minutes per transformer run on a T4)
python run_experiments.py --out results
# 2. other datasets / languages
python run_experiments.py --out results --train-on hatexplain --models bert hatebert
python run_experiments.py --out results --models xlmr
python run_experiments.py --out results --train-on hatebr --models xlmr
# 3. tables, figures and number macros for the paper
python aggregate_results.py --results results --out generated
python check_claims.py --results results         # which qualitative sentences in the paper still hold
```
Finished runs are skipped, so an interrupted run can simply be restarted. `--seeds 13 21 42` gives a quicker run.
`--smoke` runs the whole pipeline on tiny random models in seconds (a test of the code, not a result).

## Data
* Davidson et al. (2017), downloaded from the authors' repository; the SHA-256 of the file used is checked in `hs_common.py`.
* HateXplain (Mathew et al., 2021): official train/validation/test split; posts without a 2-of-3 annotator majority are dropped.
* HateBR (Vargas et al., 2022): binary offensive label only; used as an offensive-versus-not check.
Data are downloaded at run time and are not redistributed here.

## Manual step: coding the errors
`aggregate_results.py` writes `generated/errors_to_code.csv` (40 missed Hate Speech posts and 20 false alarms). Two authors
should read them independently, put one code in the `code` column and save as `generated/errors_coded.csv`
(codes: implicit_hate, sarcasm_irony, quoted_or_reported, reclaimed_ingroup, label_ambiguous, explicit_but_untargeted, other),
compute agreement with `python finalize_coding.py coder1.csv coder2.csv` (prints Cohen's kappa and writes `disagreements.csv`), resolve differences, run it again with `--resolved disagreements.csv`, and re-run `aggregate_results.py`. The paper then reports the counts.
Do not quote raw posts in the paper; describe categories instead.

## Release checklist (GitHub + Zenodo)
1. Create a public GitHub repository and push this folder (`data_cache/` is git-ignored; the raw corpora are not redistributed).
2. On zenodo.org, link your GitHub account and switch the repository on under *GitHub* settings; then create a release on GitHub. Zenodo archives it and issues a DOI (metadata is read from `.zenodo.json`).
3. Put the GitHub URL and the Zenodo DOI into `CITATION.cff` (`repository-code`, and add `doi:`) and into `\repourl` in `main.tex`.
4. `aggregate_results.py` is deterministic: the same `results/` folder and data file always give the same tables and numbers.

## Copy into the paper (checklist)
1. `python aggregate_results.py --results results --out ../paper/generated --cache data_cache` (the paper folder is next to this one).
2. `python check_claims.py --results results`. Every FAIL names the sentence in the Discussion to edit (the sentences were written
   to hold under the expected outcome, but only your run can confirm them).
3. In `paper/main.tex` replace two preamble placeholders: `\repourl` (repository URL and archive DOI) and `\errKappa`
   (Cohen's kappa of the two coders). The hardware and library line is filled automatically from the GPU run.
4. Recompile (`pdflatex`, `bibtex`, `pdflatex` twice). Read the Results and Discussion once against the final numbers.
5. Upload this folder to GitHub, archive it with Zenodo, and put the URL/DOI into `\repourl`.
6. Read `paper/refs_to_verify.txt` and spot-check the listed bibliography details.

## Note on `generated/errors_coded.csv`
The 60 sampled errors were coded once by a generative AI assistant (Claude, Anthropic), a single coder, so no inter-coder agreement exists.
Anyone can re-code them: fill your own copy of `errors_to_code.csv`, then compare with `python finalize_coding.py yours.csv generated/errors_coded.csv`.

## Quick check of the paper's numbers (no training, no data download)
`python verify_results.py` recomputes Tables 3, 4, 6, 7, the mean confusion matrices of Figure 4, the majority-class reference and the Table 9 counts
directly from `results/*/preds_test.csv` (needs only numpy, pandas, scipy). Its output for this release is in `verify_results_expected_output.txt`.
Bootstrap intervals can differ from the paper by about ±0.001 because the resampling seed differs.

## Note on post texts
The release does not contain post texts. `generated/errors_coded.csv` holds only post ids and codes; look the posts up by `id` in the public Davidson file.
`aggregate_results.py` writes `generated/errors_to_code.csv` (with post texts) when you run it locally with the data downloaded. That file is listed in `.gitignore`; do not commit it.
