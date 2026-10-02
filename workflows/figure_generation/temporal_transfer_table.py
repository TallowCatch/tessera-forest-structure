#!/usr/bin/env python3
"""Build the manuscript-ready temporal-transfer results table and preview."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/tables/dutch_temporal_transfer_summary.csv"
RATIOS = ROOT / "results/tables/dutch_temporal_transfer_rmse_ratios.csv"
OUTPUT = ROOT / "results/tables"
BUILD = ROOT / ".build/temporal_transfer_table"

OUTCOMES = [
    "p95_height_m",
    "height_sd_m",
    "height_cv",
    "entropy",
    "pulse_penetration",
    "sigma_z",
]
LABELS = {
    "p95_height_m": r"P95 height (m)",
    "height_sd_m": r"Height SD (m)",
    "height_cv": r"Height CV",
    "entropy": r"Return entropy",
    "pulse_penetration": r"Pulse penetration",
    "sigma_z": r"Sigma-z (m)",
}
CAPTION = (
    "Temporal portability of ridge models across AHN3 and AHN4. Values are "
    "medians across 15 forests. (a) Absolute-value performance for same-time "
    "AHN3 models, unchanged AHN3-to-AHN4 transfer, and contemporaneous AHN4 "
    "models. The transfer penalty is temporal-transfer RMSE divided by "
    "contemporaneous AHN4 RMSE. (b) Agreement between predicted and observed "
    "AHN4--AHN3 change. The zero-change baseline uses the AHN3 LiDAR value as "
    "the AHN4 prediction. Ratios below one favour the embedding-derived change "
    "estimate. Confidence intervals are 95\\% spatial-block bootstrap intervals "
    "across forests; RMSE is reported in each outcome's native units."
)


def metric(summary: pd.DataFrame, outcome: str, comparison: str, name: str) -> float:
    row = summary[summary["outcome"].eq(outcome) & summary["comparison"].eq(comparison)]
    if len(row) != 1:
        raise ValueError(f"Expected one {outcome}/{comparison} row, found {len(row)}")
    return float(row.iloc[0][name])


def ratio_text(ratios: pd.DataFrame, outcome: str, comparison: str) -> str:
    row = ratios[ratios["outcome"].eq(outcome) & ratios["comparison"].eq(comparison)]
    if len(row) != 1:
        raise ValueError(f"Expected one {outcome}/{comparison} row, found {len(row)}")
    values = row.iloc[0]
    return f"{values['median_rmse_ratio']:.2f} ({values['lower_95']:.2f}--{values['upper_95']:.2f})"


def build_tabular(summary: pd.DataFrame, ratios: pd.DataFrame) -> str:
    absolute_rows = []
    change_rows = []
    for outcome in OUTCOMES:
        absolute_rows.append(
            " & ".join(
                [
                    LABELS[outcome],
                    f"{metric(summary, outcome, 'same_time_ahn3', 'median_rmse'):.3f}",
                    f"{metric(summary, outcome, 'same_time_ahn3', 'median_r2'):.3f}",
                    f"{metric(summary, outcome, 'temporal_transfer_ahn4', 'median_rmse'):.3f}",
                    f"{metric(summary, outcome, 'temporal_transfer_ahn4', 'median_r2'):.3f}",
                    f"{metric(summary, outcome, 'contemporary_ahn4', 'median_rmse'):.3f}",
                    f"{metric(summary, outcome, 'contemporary_ahn4', 'median_r2'):.3f}",
                    ratio_text(ratios, outcome, "temporal_vs_contemporary"),
                ]
            )
            + r" \\"
        )
        change_rows.append(
            " & ".join(
                [
                    LABELS[outcome],
                    f"{metric(summary, outcome, 'predicted_change', 'median_rmse'):.3f}",
                    f"{metric(summary, outcome, 'zero_change', 'median_rmse'):.3f}",
                    ratio_text(ratios, outcome, "predicted_change_vs_zero"),
                    f"{metric(summary, outcome, 'predicted_change', 'median_spearman'):.3f}",
                ]
            )
            + r" \\"
        )

    return rf"""
\textbf{{(a) Absolute-value prediction}}\\[0.35em]
\begin{{tabular}}{{@{{}}lrrrrrrr@{{}}}}
\toprule
& \multicolumn{{2}}{{c}}{{Same-time AHN3}}
& \multicolumn{{2}}{{c}}{{Temporal AHN4}}
& \multicolumn{{2}}{{c}}{{Contemporaneous AHN4}}
& \multicolumn{{1}}{{c}}{{Transfer penalty}} \\
\cmidrule(lr){{2-3}} \cmidrule(lr){{4-5}} \cmidrule(lr){{6-7}} \cmidrule(l){{8-8}}
Outcome & RMSE & $R^2$ & RMSE & $R^2$ & RMSE & $R^2$ & RMSE ratio (95\% CI) \\
\midrule
{chr(10).join(absolute_rows)}
\bottomrule
\end{{tabular}}

\vspace{{0.9em}}
\textbf{{(b) AHN4--AHN3 change retrieval}}\\[0.35em]
\begin{{tabular}}{{@{{}}lrrrr@{{}}}}
\toprule
Outcome & Predicted-change RMSE & Zero-change RMSE & RMSE ratio (95\% CI) & Spearman $\rho$ \\
\midrule
{chr(10).join(change_rows)}
\bottomrule
\end{{tabular}}
""".strip()


def manuscript_table(tabular: str) -> str:
    return rf"""\begin{{table*}}[t]
\centering
\caption{{{CAPTION}}}
\label{{tab:temporal-transfer}}
\small
\setlength{{\tabcolsep}}{{4.2pt}}
{tabular}
\end{{table*}}
"""


def preview_document(tabular: str) -> str:
    return rf"""\documentclass[10pt]{{article}}
\usepackage{{booktabs}}
\usepackage{{caption}}
\usepackage[paperwidth=19cm,paperheight=13.7cm,margin=0.8cm]{{geometry}}
\pagestyle{{empty}}
\begin{{document}}
\noindent
\begin{{minipage}}{{17.3cm}}
\captionof{{table}}{{{CAPTION}}}
\vspace{{0.55em}}
\centering
\small
\setlength{{\tabcolsep}}{{4.2pt}}
{tabular}
\end{{minipage}}
\end{{document}}
"""


def main() -> None:
    summary = pd.read_csv(SUMMARY)
    ratios = pd.read_csv(RATIOS)
    tabular = build_tabular(summary, ratios)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    snippet_path = OUTPUT / "tableS_temporal_transfer.tex"
    snippet_path.write_text(manuscript_table(tabular), encoding="utf-8")

    preview_source = BUILD / "tableS_temporal_transfer_preview.tex"
    preview_source.write_text(preview_document(tabular), encoding="utf-8")
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={BUILD}",
            str(preview_source),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )
    preview_pdf = OUTPUT / "tableS_temporal_transfer_preview.pdf"
    shutil.copy2(BUILD / "tableS_temporal_transfer_preview.pdf", preview_pdf)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "220",
            "-singlefile",
            str(preview_pdf),
            str(OUTPUT / "tableS_temporal_transfer_preview"),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    print(f"Wrote {snippet_path}")
    print(f"Wrote {preview_pdf}")
    print(f"Wrote {OUTPUT / 'tableS_temporal_transfer_preview.png'}")


if __name__ == "__main__":
    main()
