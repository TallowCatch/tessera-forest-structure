#!/usr/bin/env python3
"""Create the two-page Word supervisor brief for the Cairngorms study."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/cairngorms_tessera_structural_heterogeneity_supervisor_brief.docx"
FIGURE = ROOT / "outputs/figures/cairngorms_supervisor_results_panel.png"

FONT = "Calibri"
INK = RGBColor(24, 43, 54)
TEAL = RGBColor(19, 135, 123)
BLUE = RGBColor(46, 116, 181)
MUTED = RGBColor(95, 107, 115)
LIGHT_FILL = "EEF5F4"
LIGHT_BLUE = "EEF3F9"


def set_run_font(run, *, size=None, bold=None, italic=None, color=None, name=FONT):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def set_cell_or_paragraph_shading(paragraph, fill: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = p_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        p_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_paragraph_left_border(paragraph, color: str, size: int = 18, space: int = 8) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    left = p_bdr.find(qn("w:left"))
    if left is None:
        left = OxmlElement("w:left")
        p_bdr.append(left)
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), str(size))
    left.set(qn("w:space"), str(space))
    left.set(qn("w:color"), color)


def keep_with_next(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    if p_pr.find(qn("w:keepNext")) is None:
        p_pr.append(OxmlElement("w:keepNext"))


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page ")
    set_run_font(run, size=8.5, color=MUTED)
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr, fld_char2])


def create_bullet_num_id(doc: Document) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(el.get(qn("w:abstractNumId"))) for el in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(el.get(qn("w:numId"))) for el in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)

    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "bullet")
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "\u2022")
    lvl_jc = OxmlElement("w:lvlJc")
    lvl_jc.set(qn("w:val"), "left")
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "605")
    tabs.append(tab)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "605")
    ind.set(qn("w:hanging"), "288")
    p_pr.extend([tabs, ind])
    r_pr = OxmlElement("w:rPr")
    r_fonts = OxmlElement("w:rFonts")
    r_fonts.set(qn("w:ascii"), "Arial")
    r_fonts.set(qn("w:hAnsi"), "Arial")
    r_pr.append(r_fonts)
    lvl.extend([start, num_fmt, lvl_text, lvl_jc, p_pr, r_pr])
    abstract.append(lvl)
    numbering.append(abstract)

    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    numbering.append(num)
    return num_id


def add_bullet(doc: Document, num_id: int, text: str):
    p = doc.add_paragraph(style="Brief Bullet")
    p_pr = p._p.get_or_add_pPr()
    num_pr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num_id_el = OxmlElement("w:numId")
    num_id_el.set(qn("w:val"), str(num_id))
    num_pr.extend([ilvl, num_id_el])
    p_pr.append(num_pr)
    run = p.add_run(text)
    set_run_font(run, size=10.25, color=INK)
    return p


def add_heading(doc: Document, text: str):
    p = doc.add_paragraph(style="Brief Heading")
    run = p.add_run(text)
    set_run_font(run, size=14.0, bold=True, color=BLUE)
    keep_with_next(p)
    return p


def add_callout(doc: Document, label: str, text: str, *, blue: bool = False):
    p = doc.add_paragraph(style="Brief Callout")
    set_cell_or_paragraph_shading(p, LIGHT_BLUE if blue else LIGHT_FILL)
    set_paragraph_left_border(p, "2E74B5" if blue else "13877B")
    lead = p.add_run(f"{label}  ")
    set_run_font(lead, size=10.35, bold=True, color=BLUE if blue else TEAL)
    run = p.add_run(text)
    set_run_font(run, size=10.35, color=INK)
    return p


def add_caption(doc: Document, text: str):
    p = doc.add_paragraph(style="Brief Caption")
    run = p.add_run(text)
    set_run_font(run, size=8.45, italic=True, color=MUTED)
    return p


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:ascii"), FONT)
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), FONT)
    normal.font.size = Pt(10.25)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(3.2)
    normal.paragraph_format.line_spacing = 1.05

    for style_name in ["Brief Heading", "Brief Bullet", "Brief Callout", "Brief Caption", "Brief Source"]:
        if style_name not in styles:
            styles.add_style(style_name, WD_STYLE_TYPE.PARAGRAPH)

    heading = styles["Brief Heading"]
    heading.font.name = FONT
    heading.font.size = Pt(14)
    heading.font.bold = True
    heading.font.color.rgb = BLUE
    heading.paragraph_format.space_before = Pt(8)
    heading.paragraph_format.space_after = Pt(3)
    heading.paragraph_format.keep_with_next = True

    bullet = styles["Brief Bullet"]
    bullet.font.name = FONT
    bullet.font.size = Pt(10.25)
    bullet.font.color.rgb = INK
    bullet.paragraph_format.space_before = Pt(0)
    bullet.paragraph_format.space_after = Pt(1.8)
    bullet.paragraph_format.line_spacing = 1.03

    callout = styles["Brief Callout"]
    callout.font.name = FONT
    callout.font.size = Pt(10.35)
    callout.paragraph_format.left_indent = Inches(0.12)
    callout.paragraph_format.right_indent = Inches(0.08)
    callout.paragraph_format.space_before = Pt(3)
    callout.paragraph_format.space_after = Pt(5)
    callout.paragraph_format.line_spacing = 1.05
    callout.paragraph_format.keep_together = True

    caption = styles["Brief Caption"]
    caption.font.name = FONT
    caption.font.size = Pt(8.45)
    caption.font.italic = True
    caption.font.color.rgb = MUTED
    caption.paragraph_format.space_before = Pt(2)
    caption.paragraph_format.space_after = Pt(4)
    caption.paragraph_format.line_spacing = 1.0
    caption.paragraph_format.keep_together = True

    source = styles["Brief Source"]
    source.font.name = FONT
    source.font.size = Pt(8.25)
    source.font.color.rgb = MUTED
    source.paragraph_format.space_before = Pt(0)
    source.paragraph_format.space_after = Pt(1.2)
    source.paragraph_format.line_spacing = 1.0


def build() -> None:
    if not FIGURE.exists():
        raise FileNotFoundError(f"Run make_cairngorms_supervisor_results_panel.py first: {FIGURE}")

    doc = Document()
    configure_styles(doc)
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Inches(0.58)
    section.bottom_margin = Inches(0.58)
    section.left_margin = Inches(0.67)
    section.right_margin = Inches(0.67)
    section.header_distance = Inches(0.27)
    section.footer_distance = Inches(0.28)

    header_p = section.header.paragraphs[0]
    header_p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    header_p.paragraph_format.space_after = Pt(0)
    header_run = header_p.add_run("SUPERVISOR BRIEF  |  TESSERA x CAIRNGORMS AIRBORNE LIDAR")
    set_run_font(header_run, size=8.2, bold=True, color=MUTED)
    add_page_number(section.footer.paragraphs[0])

    num_id = create_bullet_num_id(doc)

    # Page 1
    title = doc.add_paragraph()
    title.paragraph_format.space_before = Pt(7)
    title.paragraph_format.space_after = Pt(2)
    title_run = title.add_run("TESSERA embeddings for forest structural heterogeneity")
    set_run_font(title_run, size=22, bold=True, color=INK)

    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(7)
    run = subtitle.add_run("Cairngorms Connect study update  |  Ameer Alhashemi  |  31 July 2026")
    set_run_font(run, size=9.6, color=MUTED)

    add_callout(
        doc,
        "Study question",
        "How much information about airborne-LiDAR forest structure is retained in 2023 "
        "TESSERA embeddings across spatially separated mature forest in the Cairngorms?",
    )

    add_heading(doc, "What I did")
    bullets = [
        "Aligned 128-dimensional, 10 m TESSERA embeddings with a collaborator-supplied "
        "39-band airborne-LiDAR structural raster from the same 2023 Cairngorms survey.",
        "Built 26,454 non-overlapping 50 m forest units. Each unit required at least 20 valid "
        "10 m cells, mean canopy height of 10-45 m, and canopy cover of at least 0.50.",
        "Used five complete spatial test regions. For each test, the region and a 1 km buffer "
        "were excluded from training, and 1 km training blocks received equal total weight.",
        "Compared centre pixels with 30, 50 and 90 m context; gradient boosting, a two-layer "
        "MLP and a small CNN; and single-target, multi-target and vertical-profile objectives.",
        "Compared TESSERA fairly with year-matched Sentinel-1, Sentinel-2 and terrain "
        "predictors using identical 50 m support, MLP architecture, rows and test folds.",
    ]
    for item in bullets:
        add_bullet(doc, num_id, item)

    add_heading(doc, "Frozen results")
    pic_p = doc.add_paragraph()
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_p.paragraph_format.space_before = Pt(1)
    pic_p.paragraph_format.space_after = Pt(0)
    pic_p.add_run().add_picture(str(FIGURE), width=Inches(6.72))
    add_caption(
        doc,
        "Figure 1. Phase 19 out-of-fold results across five buffered spatial holdouts. "
        "Panel a separates the matched single-target comparison from the exploratory "
        "multi-target model; panels b-c show the structural signal and useful context scale.",
    )

    add_heading(doc, "What I achieved")
    for item in [
        "At matched 50 m support, TESSERA reduced canopy-entropy RMSE by 7.0% relative to "
        "Sentinel-1/2 plus terrain (R2 0.392 versus 0.298; block-bootstrap interval supported).",
        "The multi-target TESSERA MLP reached RMSE 0.102, R2 0.434 and Spearman 0.603. "
        "All five test regions had positive R2 (0.270-0.542).",
        "TESSERA was strongest for canopy gap fraction (R2 0.769) and height variation among "
        "10 m cells (R2 0.621), but weaker for a direct three-layer vertical profile (R2 0.264).",
        "A 50 m context was sufficient; 90 m added no benefit, and the ordered CNN did not "
        "improve on the compact summary MLP.",
    ]:
        add_bullet(doc, num_id, item)

    doc.add_page_break()

    # Page 2
    add_heading(doc, "Interpretation")
    for item in [
        "The embeddings contain useful information about forest structure beyond ordinary "
        "annual satellite and terrain predictors, especially canopy openness and horizontal "
        "variation in canopy height.",
        "Prediction of the supplied canopy-return Shannon entropy is moderate rather than "
        "operational. The pre-declared R2 usefulness target of 0.70 was not reached.",
        "The current evidence supports a within-Cairngorms representation study. It does not "
        "yet support reliable transfer to another landscape or a national structural map.",
    ]:
        add_bullet(doc, num_id, item)

    add_heading(doc, "Literature read and how it changed the design")
    literature = [
        "Feng et al., TESSERA (arXiv:2506.20380): established the embedding and canopy-height "
        "benchmark; height prediction alone does not answer the heterogeneity question.",
        "Atkins et al. (2023), Scale dependency of lidar-derived forest structural diversity: "
        "motivated testing 10-90 m context and selecting 50 m as the main ecological grain.",
        "Roberts et al. (2017) and Ploton et al. (2020): motivated complete spatial holdouts and "
        "buffers instead of random train-test splits that can exploit nearby observations.",
        "Rosen et al. (2024), structural complexity through space and time: showed why one "
        "index is insufficient, motivating separate vertical, horizontal and gap measurements.",
        "Fischer et al. (2024): highlighted how airborne-LiDAR processing choices affect "
        "structural metrics and made exact target provenance a necessary limitation.",
        "Davison et al. (2023): supplied the ecological rationale for FHD and canopy-height "
        "variation as biodiversity-relevant measurements.",
        "Thapa and Chan draft on Cairngorms woodland expansion, and Khan et al. (2026) FSKD: "
        "provided local LiDAR context and a larger deep-learning comparison for future work.",
    ]
    for item in literature:
        add_bullet(doc, num_id, item)

    add_heading(doc, "Current limitations")
    for item in [
        "The supplied Shannon band is bounded by ln(5), indicating raw five-bin entropy, while "
        "the available lidarSHM documentation describes normalization and filtering above "
        "1.3 m. The exact historical production settings remain unconfirmed.",
        "The 50 m target averages 25 supplied 10 m entropy cells; it is not entropy recomputed "
        "from pooled LiDAR returns over a 50 m footprint.",
        "Repeated work on the same five folds makes this a model-development analysis. A new, "
        "same-processed landscape is required for an untouched replication.",
        "Gap and cover are almost exact inverses, and height metrics are correlated. The final "
        "paper should report a small non-redundant target set.",
    ]:
        add_bullet(doc, num_id, item)

    add_heading(doc, "Open questions for supervision")
    for item in [
        "Should I ask Aland to confirm the exact point-height threshold, binning and "
        "normalization used for the supplied Shannon raster, or should raw point clouds be "
        "reprocessed before manuscript submission?",
        "Is the strongest paper framing an evaluation of what TESSERA represents about forest "
        "structure, rather than a claim that it can already generate an operational FHD map?",
        "Which independent Scottish landscape can be processed identically for a frozen "
        "replication? The current Arran transfer result is provisional because year and LiDAR "
        "normalization differ.",
        "Should the final ecological panel focus on canopy entropy, between-cell height "
        "variation, gap fraction and the three-layer vertical profile?",
    ]:
        add_bullet(doc, num_id, item)

    add_callout(
        doc,
        "Recommended next step",
        "Freeze the 50 m multi-target MLP, stop broad architecture searches, resolve the "
        "Shannon-target provenance, and run one genuinely independent same-processing "
        "replication before deciding whether a cautious prediction map is warranted.",
        blue=True,
    )

    sources = doc.add_paragraph(style="Brief Source")
    lead = sources.add_run("Key references: ")
    set_run_font(lead, size=8.25, bold=True, color=MUTED)
    text = (
        "Feng et al., TESSERA, arXiv:2506.20380; Atkins et al. 2023, Methods in Ecology "
        "and Evolution, doi:10.1111/2041-210X.14040; Roberts et al. 2017, Ecography, "
        "doi:10.1111/ecog.02881; Ploton et al. 2020, Nature Communications, "
        "doi:10.1038/s41467-020-18321-y; Rosen et al. 2024, Ecography, "
        "doi:10.1111/ecog.07377; Fischer et al. 2024, Methods in Ecology and Evolution, "
        "doi:10.1111/2041-210X.14416; Davison et al. 2023, Journal of Animal Ecology, "
        "doi:10.1111/1365-2656.13945; Khan et al. 2026, arXiv:2604.01766."
    )
    run = sources.add_run(text)
    set_run_font(run, size=8.25, color=MUTED)

    # Avoid carrying a blank trailing section or default paragraph.
    for p in doc.paragraphs:
        if not p.text and not p._p.xpath(".//w:drawing") and p is not doc.paragraphs[-1]:
            continue

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
