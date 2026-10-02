#!/usr/bin/env python3
"""Build a concise meeting packet from the current-analysis audit artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/Ameer_premeeting_current_analysis_audit.pdf"
FIGURES = ROOT / "outputs/figures"
TABLES = ROOT / "outputs/tables"


def page_number(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(
        landscape(A4)[0] - 15 * mm,
        9 * mm,
        f"Ameer Alhashemi | Pre-meeting analysis audit | {document.page}",
    )
    canvas.restoreState()


def table_from_frame(frame: pd.DataFrame, widths: list[float]) -> Table:
    values = [list(frame.columns)] + frame.astype(str).values.tolist()
    table = Table(values, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8ECEB")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#202020")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.7, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def fit_image(path: Path, max_width: float, max_height: float) -> Image:
    image = Image(str(path))
    ratio = min(max_width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def figure_page(
    story: list,
    title: str,
    path: Path,
    caption: str,
    heading_style: ParagraphStyle,
    caption_style: ParagraphStyle,
) -> None:
    story.append(Paragraph(title, heading_style))
    story.append(Spacer(1, 3 * mm))
    story.append(fit_image(path, 257 * mm, 145 * mm))
    story.append(Spacer(1, 2.5 * mm))
    story.append(Paragraph(caption, caption_style))
    story.append(PageBreak())


def run() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = landscape(A4)
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title="Pre-meeting audit of the current forest-heterogeneity analyses",
        author="Ameer Alhashemi",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "AuditTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=25,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1B2927"),
        spaceAfter=6 * mm,
    )
    heading_style = ParagraphStyle(
        "AuditHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=17,
        textColor=colors.HexColor("#1B2927"),
        spaceAfter=2 * mm,
    )
    body_style = ParagraphStyle(
        "AuditBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#202020"),
        spaceAfter=2.5 * mm,
    )
    caption_style = ParagraphStyle(
        "AuditCaption",
        parent=body_style,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#444444"),
    )
    bullet_style = ParagraphStyle(
        "AuditBullet",
        parent=body_style,
        leftIndent=5 * mm,
        firstLineIndent=-3.5 * mm,
        bulletIndent=0,
        spaceAfter=1.8 * mm,
    )

    story: list = []
    story.append(Paragraph("Pre-meeting audit of the current analyses", title_style))
    story.append(
        Paragraph(
            "Cairngorms canopy heterogeneity, canopy height, Savelsbos replication, "
            "cross-landscape transfer and two-landscape pooled training",
            heading_style,
        )
    )
    story.append(Spacer(1, 5 * mm))
    for text in [
        "The audit independently recomputed 1,565 saved metric values from the held-out prediction files. All matched their result tables within numerical precision.",
        "The five Cairngorms test folds are disjoint, cover all 17,316 units exactly once and retain a minimum train-test separation of 2,000.6 m.",
        "All 58 Cairngorms prediction groups contain complete, unique and finite held-out predictions. Frozen prediction files and numerical tables retain their recorded hashes.",
        "The high canopy-heterogeneity scores are therefore internally reproducible. The diagnostics below show where those summary scores hide range compression, spatial residual structure and dependence on observed canopy height.",
    ]:
        story.append(Paragraph(text, bullet_style, bulletText="•"))
    story.append(Spacer(1, 6 * mm))

    summary = pd.DataFrame(
        [
            ["Height CV", "0.798", "0.324", "0.931"],
            ["Canopy openings", "0.786", "0.344", "0.868"],
            ["Rumple", "0.586", "0.648", "-0.003"],
            ["Height SD", "0.651", "0.342", "0.809"],
            ["Robust CV", "0.676", "0.140", "0.847"],
            ["Height kurtosis", "0.610", "0.185", "0.624"],
        ],
        columns=[
            "Cairngorms outcome",
            "Raw TESSERA v2 R2",
            "Height-adjusted R2",
            "Variance removed by observed height",
        ],
    )
    story.append(Paragraph("Height-adjustment summary", heading_style))
    story.append(table_from_frame(summary, [55 * mm, 40 * mm, 40 * mm, 67 * mm]))
    story.append(PageBreak())

    story.append(Paragraph("Audit findings that need a decision", heading_style))
    findings = [
        "<b>Height adjustment is explanatory.</b> It uses observed LiDAR mean and p95 height. It tests predictability conditional on known stature; it is not itself a wall-to-wall prediction workflow.",
        "<b>Raw and adjusted R2 cannot be treated as a variance partition.</b> The response and its variance change. The decomposition figure therefore reports both the variance removed and the residual-model score.",
        "<b>The tallest trees remain compressed.</b> In the tallest p95 decile, mean observed height is 25.49 m; the MLP and strict U-Net predict 22.20 m and 22.17 m, respectively.",
        "<b>SD/CV use N-1 in the current target builder.</b> Population SD is clearer for complete 50 m units. Correcting N-1 to N changes values by at most 0.0201%, so this is a definition/reproducibility issue rather than an explanation for the results.",
        "<b>Rank metrics differ between landscapes.</b> Cairngorms correlates 1 km block means; Savelsbos averages correlations calculated within 500 m blocks. They require distinct labels or harmonisation before direct comparison.",
        "<b>Pooling two landscapes is not the proposed multi-site solution.</b> The joint MLP is slightly worse than local training, but two strongly different sites cannot test whether broader multi-site training would learn a stable relationship.",
    ]
    for text in findings:
        story.append(Paragraph(text, bullet_style, bulletText="•"))
    story.append(Spacer(1, 3 * mm))
    formula = pd.read_csv(TABLES / "premeeting_target_formula_audit.csv")
    formula = formula[
        ["target", "status", "maximum_relative_change_percent"]
    ].copy()
    formula.columns = ["Target", "Audit status", "Maximum change (%)"]
    formula["Maximum change (%)"] = formula["Maximum change (%)"].map(
        lambda value: f"{value:.4f}"
    )
    story.append(KeepTogether([Paragraph("Target-formula checks", heading_style), table_from_frame(formula, [55 * mm, 137 * mm, 36 * mm])]))
    story.append(PageBreak())

    figure_page(
        story,
        "Held-out canopy-heterogeneity fits",
        FIGURES / "premeeting_cairngorms_surface_observed_predicted.png",
        "Observed-versus-predicted plots for the six principal Cairngorms canopy-heterogeneity outcomes. The dashed line is exact agreement. High R2 coexists with regression toward the mean at distribution extremes.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "Height-adjusted canopy-heterogeneity fits",
        FIGURES / "premeeting_cairngorms_height_adjusted_observed_predicted.png",
        "The response is the fold-specific residual after fitting each structural outcome from observed LiDAR mean and p95 height in the training data. Rumple retains the clearest residual relationship.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "Canopy-height fits and model comparison",
        FIGURES / "premeeting_cairngorms_height_observed_predicted.png",
        "The strict U-Net improves mean and p95 height relative to the 5 x 5 MLP, but both models compress the observed height range.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "Canopy-height range compression",
        FIGURES / "premeeting_cairngorms_height_decile_diagnostic.png",
        "Observed and predicted means within observed-height deciles. Underprediction grows in the upper tail, while the shortest units are overpredicted.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "What the observed-height adjustment removes",
        FIGURES / "premeeting_height_adjustment_decomposition.png",
        "CV, openings, robust CV and height SD are strongly related to observed LiDAR stature. Rumple is the exception: observed height alone transfers poorly, whereas TESSERA retains substantial predictive performance for the residual outcome.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "Cairngorms target distributions",
        FIGURES / "premeeting_cairngorms_target_distributions.png",
        "Distributions after all cohort filters. Several outcomes are strongly skewed, so fit panels and spatially held-out evaluation are necessary alongside mean R2.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "Spatial residual structure",
        FIGURES / "premeeting_cairngorms_spatial_residuals.png",
        "Held-out residuals remain geographically patterned. The fold design prevents these local patterns from being mistaken for random independent errors, but the patterns should be discussed when interpreting landscape-wide performance.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "Cross-landscape differences in observed outcomes",
        FIGURES / "premeeting_cross_site_target_distributions.png",
        "The four approximately harmonised targets occupy different distributions in the Cairngorms and Savelsbos. This is consistent with domain shift but does not isolate forest type from LiDAR product and acquisition differences.",
        heading_style,
        caption_style,
    )
    figure_page(
        story,
        "Direct transfer and two-landscape joint training",
        FIGURES / "premeeting_transfer_and_pooling_diagnostic.png",
        "Bars show RMSE relative to a locally trained model on the same held-out areas. Zero-shot transfer fails strongly. Joint MLP training remains close to, but generally worse than, local training.",
        heading_style,
        caption_style,
    )

    document.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(OUTPUT)


if __name__ == "__main__":
    run()
