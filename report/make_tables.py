"""Generate LaTeX tables for the report directly from the saved results files.

Numbers in the report tables are therefore never typed by hand.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)


def f(x, d=3):
    return f"{x:.{d}f}"


def target_table():
    ver = json.loads((ROOT / "data" / "interim" / "brfss2015_subset.verification.json").read_text())
    obs = ver["diabete3_counts"]
    labels = {"1.0": ("Yes", "Diabetes"), "2.0": ("Yes, only during pregnancy", "No diabetes"),
              "3.0": ("No", "No diabetes"), "4.0": ("No, pre-/borderline diabetes", "Prediabetes"),
              "7.0": ("Don't know", "removed"), "9.0": ("Refused", "removed"), "missing": ("Blank", "removed")}
    rows = "\n".join(f"{k.replace('.0', '')} & {labels[k][0]} & {int(v):,} & {labels[k][1]} \\\\" for k, v in obs.items())
    (OUT / "target.tex").write_text(rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{\texttt{{DIABETE3}} categories, observed counts (identical to the codebook~\cite{{cdc2015codebook}}) and the 3-class target mapping.}}
\label{{tab:target}}
\begin{{tabular}}{{@{{}}clrl@{{}}}}
\toprule
Code & Response & $n$ & Target class\\
\midrule
{rows}
\midrule
 & Total & {ver['n_records']:,} & \\
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def experiments_table():
    r = pd.read_csv(REP / "experiment_results.csv")
    short = {
        "E01": "LR baseline", "E02": "LR unscaled", "E03": "LR balanced", "E04": "RF baseline",
        "E05": "GB baseline", "E06": "GB + engineered", "E07": "GB tuned", "E08": "GB tuned, balanced",
        "E09": "RF tuned", "E10": "MLP 32 (Seq.)", "E11": "MLP 128-64-32 reg. (Seq.)", "E12": "E11, lr $10^{-2}$",
        "E13": "E11, lr $10^{-4}$", "E14": "E11, batch 4096", "E15": "E11, class-weighted",
        "E16": "Wide-and-deep (Func.)", "E17": "E16 + engineered"}
    best = r["avg_precision_macro"].max()
    lines = []
    for _, x in r.iterrows():
        ap = f(x.avg_precision_macro, 4)
        if x.avg_precision_macro == best:
            ap = r"\textbf{" + ap + "}"
        feat = "B+E" if "engineered" in x.features else "B"
        imb = "CW" if x.imbalance_strategy != "none" else "--"
        lines.append(f"{x.experiment_id} & {short[x.experiment_id]} & {feat} & {imb} & {x.train_seconds:.0f} & "
                     f"{f(x.accuracy)} & {f(x.f1_macro)} & {f(x.roc_auc_ovr_macro, 4)} & {ap} & "
                     f"{f(x.recall_prediabetes, 2)} & {f(x.recall_diabetes, 2)} \\\\")
    body = "\n".join(lines)
    (OUT / "experiments.tex").write_text(rf"""\begin{{table*}}[t]
\centering\footnotesize
\caption{{Experiment results on the validation set (66,099 respondents; argmax decisions). Split: stratified 70/15/15, seed 42. B = 25 base features, B+E = base + 6 engineered; CW = class weighting. Time = training seconds on 2 CPU cores. Macro AP is the selection metric; the best value is in bold. Full hyper-parameters are in Appendix~\ref{{app:configs}} and \texttt{{reports/experiment\_log.jsonl}}.}}
\label{{tab:experiments}}
\begin{{tabular}}{{@{{}}llccrcccccc@{{}}}}
\toprule
ID & Model & Feat. & Imb. & Time (s) & Acc. & Macro F1 & Macro ROC-AUC & Macro AP & Rec.\ pre. & Rec.\ dia.\\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")


def test_table():
    t = pd.read_csv(REP / "test_results.csv", index_col=0)
    cols = [("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("f1_macro", "Macro F1"),
            ("roc_auc_ovr_macro", "Macro ROC-AUC"), ("avg_precision_macro", "Macro AP"),
            ("recall_prediabetes", "Recall, prediabetes"), ("precision_prediabetes", "Precision, prediabetes"),
            ("recall_diabetes", "Recall, diabetes"), ("precision_diabetes", "Precision, diabetes"),
            ("weighted_f1_macro", "Macro F1 (survey-weighted)"), ("weighted_roc_auc", "Macro ROC-AUC (survey-weighted)"),
            ("weighted_recall_diabetes", "Recall, diabetes (survey-weighted)")]
    names = [n.split(" ", 1)[0] for n in t.index]
    header = " & ".join(["Metric"] + [rf"\textbf{{{n}}}" for n in names])
    rows = []
    for c, label in cols:
        vals = t[c].tolist()
        best = max(vals)
        cells = [(r"\textbf{" + f(v) + "}") if v == best else f(v) for v in vals]
        rows.append(" & ".join([label] + cells) + r" \\")
    (OUT / "test.tex").write_text(rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Final held-out test results (66,099 respondents) with validation-tuned decision weights. E01 = logistic regression, E07 = tuned gradient boosting, E16 = Functional wide-and-deep network. Best value per row in bold.}}
\label{{tab:test}}
\begin{{tabular}}{{@{{}}lccc@{{}}}}
\toprule
{header}\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def screening_table():
    s = pd.read_csv(REP / "screening_results.csv", index_col=0)
    lab = {"default_0.5": "Default 0.5", "F2_threshold": "$F_2$-optimal", "F2_threshold_survey_weighted": "$F_2$-optimal, weighted"}
    rows = "\n".join(
        f"{lab[i]} & {f(x.threshold)} & {f(x.sensitivity)} & {f(x.specificity)} & {f(x.ppv)} & {f(x.npv)} & {f(x.flag_rate)} \\\\"
        for i, x in s.iterrows())
    (OUT / "screening.tex").write_text(rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Binary screening view (diabetes vs.\ not) for tuned gradient boosting on the test set. The $F_2$ threshold was selected on validation.}}
\label{{tab:screening}}
\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\toprule
Rule & Thr. & Sens. & Spec. & PPV & NPV & Flagged\\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def subgroup_table():
    s = pd.read_csv(REP / "subgroup_metrics.csv")
    rows = "\n".join(
        f"{x.dimension} & {str(x.group).replace(chr(36), chr(92) + chr(36))} & {int(x.n):,} & {f(x.diabetes_prevalence)} & {f(x.recall_diabetes)} & "
        f"{f(x.precision_diabetes)} & {f(x.roc_auc_diabetes)} \\\\" for _, x in s.iterrows())
    (OUT / "subgroups.tex").write_text(rf"""\begin{{table}}[t]
\centering\footnotesize
\caption{{Test-set diabetes detection by subgroup (tuned gradient boosting, tuned decision weights). Race/ethnicity was not a model input. All subgroups have at least 163 diabetes cases.}}
\label{{tab:subgroups}}
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}llrcccc@{{}}}}
\toprule
Dim. & Group & $n$ & Prev. & Recall & Prec. & AUC\\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def esc(t):
    t = str(t)
    for a, b in [("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("$", r"\$"), ("^", r"\^{}"), ("{", r"\{"), ("}", r"\}")]:
        t = t.replace(a, b)
    return t


def config_table():
    lines = []
    for line in (REP / "experiment_log.jsonl").read_text().splitlines():
        r = json.loads(line)
        hp = r["config"]["hyperparameters"]
        hp = {k: v for k, v in hp.items() if k not in ("search",)}
        txt = esc(", ".join(f"{k}={v}" for k, v in hp.items()))
        lines.append(f"{r['experiment_id']} & \\texttt{{{esc(r['run_id'])}}} & {txt} & {esc(r.get('observation', ''))} \\\\")
    (OUT / "configs.tex").write_text(r"""\begin{footnotesize}
\begin{longtable}{@{}p{0.04\linewidth}p{0.17\linewidth}p{0.41\linewidth}p{0.32\linewidth}@{}}
\toprule
ID & Run ID & Hyper-parameters & Observation (after reading the results)\\
\midrule
\endhead
""" + "\n".join(lines) + "\n\\bottomrule\n\\end{longtable}\n\\end{footnotesize}\n")


if __name__ == "__main__":
    target_table()
    experiments_table()
    test_table()
    screening_table()
    subgroup_table()
    config_table()
    print("tables written to", OUT)
