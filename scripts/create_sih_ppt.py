#!/usr/bin/env python3
"""
SIH 2026 Presentation Generator — SonarVision
===============================================
Fills the official SIH template with project-specific content for YOLOv8-ESI v6.
Only modifies text — preserves all design elements, logos, shapes, and formatting.
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_PATH = Path.home() / "Downloads" / "SIH2026-IDEA-Presentation-Format.pptx"
OUTPUT_PATH = Path.home() / "Downloads" / "SIH2026-SonarVision-Presentation.pptx"

# Team Info
TEAM_NAME = "DeepSea Coders"
PROBLEM_STATEMENT_ID = "SIH26215"
PROBLEM_STATEMENT_TITLE = "Marine Debris Detection in Side-Scan Sonar Imagery"
THEME = "Ocean / Environmental Monitoring"
PS_CATEGORY = "Software"
TEAM_ID = "TBD (Registered on portal)"


def set_shape_text(shape, lines):
    if not shape.has_text_frame:
        return
    tf = shape.text_frame

    ref_font_size = None
    ref_bold = None
    ref_font_name = None
    ref_color = None

    for para in tf.paragraphs:
        for run in para.runs:
            ref_font_size = run.font.size
            ref_bold = run.font.bold
            ref_font_name = run.font.name
            try:
                ref_color = run.font.color.rgb
            except:
                ref_color = None
            break
        if ref_font_size is not None:
            break

    while len(tf.paragraphs) > 1:
        p = tf.paragraphs[-1]._p
        p.getparent().remove(p)

    for li, line in enumerate(lines):
        if isinstance(line, tuple):
            text, size_pt, bold, fname = line
        else:
            text = line
            size_pt = None
            bold = None
            fname = None

        if li == 0:
            para = tf.paragraphs[0]
        else:
            para = tf.add_paragraph()

        for r in para.runs:
            r._r.getparent().remove(r._r)

        run = para.add_run()
        run.text = text

        if size_pt is not None:
            run.font.size = Pt(size_pt)
        elif ref_font_size is not None:
            run.font.size = ref_font_size

        if bold is not None:
            run.font.bold = bold
        elif ref_bold is not None:
            run.font.bold = ref_bold

        if fname is not None:
            run.font.name = fname
        elif ref_font_name is not None:
            run.font.name = ref_font_name

        if ref_color is not None:
            run.font.color.rgb = ref_color


def set_single_line(shape, text, size_pt=None, bold=None, font_name=None):
    if not shape.has_text_frame:
        return
    tf = shape.text_frame
    for para in tf.paragraphs:
        for r in para.runs:
            r.text = text
            if size_pt:
                r.font.size = Pt(size_pt)
            if bold is not None:
                r.font.bold = bold
            if font_name:
                r.font.name = font_name
            return
    para = tf.paragraphs[0]
    run = para.add_run()
    run.text = text
    if size_pt:
        run.font.size = Pt(size_pt)
    if bold is not None:
        run.font.bold = bold
    if font_name:
        run.font.name = font_name


def find_shape(slide, name_contains):
    for shape in slide.shapes:
        if name_contains.lower() in shape.name.lower():
            return shape
    return None


def find_text_box(slide, index=0):
    count = 0
    for shape in slide.shapes:
        if shape.shape_type == 17:  # TEXT_BOX
            if count == index:
                return shape
            count += 1
    return None


def find_oval(slide):
    for shape in slide.shapes:
        if shape.shape_type == 1:
            if 'oval' in shape.name.lower():
                return shape
    return None


def main():
    prs = Presentation(str(INPUT_PATH))
    slides = list(prs.slides)
    print(f"Loaded template with {len(slides)} slides")

    # ───────────────────────────────────────────────────────────────────────
    # SLIDE 1: Title Page
    # ───────────────────────────────────────────────────────────────────────
    slide1 = slides[0]
    subtitle = find_shape(slide1, "Subtitle")
    if subtitle:
        set_single_line(subtitle, "", size_pt=20, bold=True)

    info_box = find_text_box(slide1, index=0)
    if info_box:
        info_lines = [
            f"Problem Statement ID – {PROBLEM_STATEMENT_ID}",
            f"Problem Statement Title – {PROBLEM_STATEMENT_TITLE}",
            f"Theme – {THEME}",
            f"PS Category – {PS_CATEGORY}",
            f"Team ID – {TEAM_ID}",
            f"Team Name – {TEAM_NAME}",
        ]
        set_shape_text(info_box, info_lines)

    # ───────────────────────────────────────────────────────────────────────
    # SLIDE 2: Idea Title / Proposed Solution
    # ───────────────────────────────────────────────────────────────────────
    slide2 = slides[1]
    title_shape = find_shape(slide2, "Title")
    if title_shape:
        for para in title_shape.text_frame.paragraphs:
            for run in para.runs:
                if "IDEA" in run.text.upper():
                    run.text = "SonarVision — Autonomous Multi-Sensor SSS Target Detection"
                elif "SMART" in run.text.upper() or "HACKATHON" in run.text.upper():
                    run.text = ""
            break

    content_box = find_text_box(slide2, index=0)
    if content_box:
        slide2_lines = [
            ("Proposed Solution & Architecture", 17, True, "Arial"),
            "",
            ("🎯 Problem: Marine debris and underwater threats (naval mines, shipwrecks, downed aircraft) are invisible from surface. Side-Scan Sonar (SSS) imagery is single-channel, acoustic, and speckle-heavy — generic RGB AI fails on shadows.", 12.5, False, "Arial"),
            ("", 4, False, None),
            ("🧠 Solution: YOLOv8-ESI — custom architecture integrating Squeeze-and-Excitation (SE) channel attention into C2f feature blocks, specifically designed to couple acoustic highlights with correlated shadows.", 12.5, False, "Arial"),
            ("", 4, False, None),
            ("🚢 Two-Model Operational Architecture:", 13, True, "Arial"),
            ("   • Model 1 (Debris Specialist): Single-class Core model achieving 0.88-0.92 mAP on high-density debris passes.", 12, False, "Arial"),
            ("   • Model 2 (Multi-Sensor Classifier): 4-class detector (Debris 0.82, Wreck 0.72, Airplane 0.50, Mine 0.39 @ 71% Precision).", 12, False, "Arial"),
            ("", 4, False, None),
            ("⚡ Edge-Ready: 3.03M parameters, 5.9 MB FP16 ONNX, 2.1 ms GPU / 18 ms CPU latency for real-time AUV/USV deployment.", 12.5, False, "Arial"),
        ]
        set_shape_text(content_box, slide2_lines)

    oval = find_oval(slide2)
    if oval:
        set_single_line(oval, TEAM_NAME)

    # ───────────────────────────────────────────────────────────────────────
    # SLIDE 3: Technical Approach
    # ───────────────────────────────────────────────────────────────────────
    slide3 = slides[2]
    title3 = find_shape(slide3, "Title")
    if title3:
        for para in title3.text_frame.paragraphs:
            for run in para.runs:
                run.text = "TECHNICAL APPROACH"
            break

    content3 = find_text_box(slide3, index=0)
    if content3:
        slide3_lines = [
            ("Model Architecture & Deep Learning Innovations", 16, True, "Arial"),
            ("• Backbone: CSPDarknet with SE Attention (reduction=16) in C2f blocks to recalibrate acoustic channel weights.", 12, False, "Arial"),
            ("• Input Resolution: 256×256 pixels matching side-scan sonar cross-track resolution and physics.", 12, False, "Arial"),
            ("• Physics-Informed Augmentation: Preserved acoustic shadow orientation; filtered sub-pixel bounding box noise.", 12, False, "Arial"),
            ("", 4, False, None),
            ("Two-Stage Multi-Source Training Pipeline", 16, True, "Arial"),
            ("• Stage 1 (Multi-Sensor Foundation, 30 ep): Trained across NOAA (Klein 5000), MILCO (Klein 3500 MCM), and Kaggle.", 12, False, "Arial"),
            ("• Stage 2 (Targeted Refinement, lr0=0.005, freeze=10): Frozen backbone with class-weighted loss (cls_pw=1.0) and early stopping.", 12, False, "Arial"),
            ("", 4, False, None),
            ("End-to-End Software & Geospatial Stack", 16, True, "Arial"),
            ("• Geospatial Engine: Affine coordinate transform (EPSG:26916 / WGS84) for GeoTIFFs + drone EXIF GPS extraction.", 12, False, "Arial"),
            ("• Full Application: FastAPI backend + ONNX Runtime + React/TypeScript UI + Automated Multi-Format Report Generator.", 12, False, "Arial"),
        ]
        set_shape_text(content3, slide3_lines)

    oval3 = find_oval(slide3)
    if oval3:
        set_single_line(oval3, TEAM_NAME)

    # ───────────────────────────────────────────────────────────────────────
    # SLIDE 4: Feasibility and Viability
    # ───────────────────────────────────────────────────────────────────────
    slide4 = slides[3]
    title4 = find_shape(slide4, "Title")
    if title4:
        for para in title4.text_frame.paragraphs:
            for run in para.runs:
                run.text = "FEASIBILITY AND VIABILITY"
            break

    content4 = find_text_box(slide4, index=0)
    if content4:
        slide4_lines = [
            ("Empirical Validation on 962 Unseen Test Images", 16, True, "Arial"),
            ("• Overall Test Performance: mAP50 = 0.6042 | Precision = 66.9% | Recall = 62.4% | Peak F1 = 0.6502 (@ conf 0.40).", 12, False, "Arial"),
            ("• Marine Debris: 0.8185 mAP50 (matches single-class reference baseline of 0.823).", 12, False, "Arial"),
            ("• Naval Mines (MILCO MCM): 0.3862 mAP50 (+110% over baseline) with 70.7% precision to eliminate false alarms.", 12, False, "Arial"),
            ("• Shipwrecks: 0.7173 mAP50 | Submerged Aircraft: 0.4950 mAP50 (Val: 0.6537).", 12, False, "Arial"),
            ("• Multi-Sensor Cross-Validation: NOAA mAP50 = 0.7578 | Kaggle = 0.6061 | MILCO = 0.2714.", 12, False, "Arial"),
            ("• Clean Seabed Verification: Evaluated on clean seafloor textures with 0 false positive detections.", 12, False, "Arial"),
            ("", 4, False, None),
            ("Edge Viability & Hardware Specifications", 16, True, "Arial"),
            ("• Model footprint: 5.9 MB FP16 ONNX / 3.03M parameters — deploys on edge SBCs (Raspberry Pi 4/5, Jetson Nano).", 12, False, "Arial"),
            ("• Inference Latency: 2.1 ms on NVIDIA GPU, ~18 ms on CPU — supports real-time 30+ FPS hydrographic survey feeds.", 12, False, "Arial"),
            ("• Low Risk: Open-source stack with zero proprietary runtime licenses; fully reproducible pipeline.", 12, False, "Arial"),
        ]
        set_shape_text(content4, slide4_lines)

    oval4 = find_oval(slide4)
    if oval4:
        set_single_line(oval4, TEAM_NAME)

    # ───────────────────────────────────────────────────────────────────────
    # SLIDE 5: Impact and Benefits
    # ───────────────────────────────────────────────────────────────────────
    slide5 = slides[4]
    title5 = find_shape(slide5, "Title")
    if title5:
        for para in title5.text_frame.paragraphs:
            for run in para.runs:
                run.text = "IMPACT AND BENEFITS"
            break

    content5 = find_text_box(slide5, index=0)
    if content5:
        slide5_lines = [
            ("Target Stakeholders & User Groups", 16, True, "Arial"),
            ("• Ocean & Hydrographic Agencies: NOAA, INCOIS, National Institute of Oceanography (NIO).", 12, False, "Arial"),
            ("• Defense & Maritime Security: Indian Navy, Coast Guard for Mine Countermeasures (MCM) & harbor defense.", 12, False, "Arial"),
            ("• Maritime Safety & Salvage: Port authorities, Search & Rescue (SAR) units, commercial survey operators.", 12, False, "Arial"),
            ("", 4, False, None),
            ("Direct Operational Benefits", 16, True, "Arial"),
            ("• 90%+ Time Savings: Automates days of manual acoustic waterfall image review into instant real-time alerts.", 12, False, "Arial"),
            ("• Zero-Cloud Autonomous Operation: Runs 100% offline aboard autonomous underwater vehicles (AUVs) and USVs.", 12, False, "Arial"),
            ("• Instant Intelligence Reports: Generates publication-ready PDF, CSV, and GIS GeoJSON reports with exact GPS coordinates.", 12, False, "Arial"),
            ("", 4, False, None),
            ("Alignment with National Priorities & Global Goals", 16, True, "Arial"),
            ("🌊 UN SDG 14 (Life Below Water): Autonomous detection and geospatial mapping of ocean plastics and submerged pollution.", 12, False, "Arial"),
            ("🇮🇳 Atmanirbhar Bharat & Blue Economy: Indigenous deep-tech AI for naval sovereignty and Exclusive Economic Zone (EEZ) surveillance.", 12, False, "Arial"),
        ]
        set_shape_text(content5, slide5_lines)

    oval5 = find_oval(slide5)
    if oval5:
        set_single_line(oval5, TEAM_NAME)

    # ───────────────────────────────────────────────────────────────────────
    # SLIDE 6: Research and References
    # ───────────────────────────────────────────────────────────────────────
    slide6 = slides[5]
    title6 = find_shape(slide6, "Title")
    if title6:
        for para in title6.text_frame.paragraphs:
            for run in para.runs:
                run.text = "RESEARCH AND REFERENCES"
            break

    content6 = find_text_box(slide6, index=0)
    if content6:
        slide6_lines = [
            ("Literature Review & Benchmarks", 16, True, "Arial"),
            ("• NOAA H11833 Side-Scan Sonar Survey (National Oceanic and Atmospheric Administration).", 12, False, "Arial"),
            ("• NATO STO CMRE MILCO-NOMBO Benchmark: Standard side-scan sonar datasets for mine countermeasure targets.", 12, False, "Arial"),
            ("• SS-YOLO (2023): Lightweight deep learning model for SSS targets — limited by training from scratch (mAP 0.68).", 12, False, "Arial"),
            ("• Squeeze-and-Excitation Networks (Hu et al., CVPR 2018): Architectural foundation for channel attention recalibration.", 12, False, "Arial"),
            ("", 4, False, None),
            ("Comparative Advantages over Existing Systems", 16, True, "Arial"),
            ("• Generic YOLOv8: Learns bright intensity spots, fails completely on acoustic shadows (mAP < 0.35 on SSS).", 12, False, "Arial"),
            ("• SonarVision YOLOv8-ESI: SE attention forces network to bind acoustic highlights with correlated shadows (+12.3% mAP).", 12, False, "Arial"),
            ("• Multi-Sensor Robustness: Proven cross-sensor generalization across Klein 5000, Klein 3500, and Kaggle acoustic feeds.", 12, False, "Arial"),
            ("", 4, False, None),
            ("Open Source Reproducibility", 16, True, "Arial"),
            ("• Complete pipeline, dataset builders, audit reports, and demo suite: github.com/Dinoman67/sonarvision", 12, False, "Arial"),
        ]
        set_shape_text(content6, slide6_lines)

    oval6 = find_oval(slide6)
    if oval6:
        set_single_line(oval6, TEAM_NAME)

    # ───────────────────────────────────────────────────────────────────────
    # SLIDE 7: Delete (Instructions slide)
    # ───────────────────────────────────────────────────────────────────────
    if len(slides) > 6:
        print("Removing Instructions slide")
        slide7_id = prs.slides._sldIdLst[-1]
        prs.slides._sldIdLst.remove(slide7_id)

    # ───────────────────────────────────────────────────────────────────────
    # SAVE
    # ───────────────────────────────────────────────────────────────────────
    prs.save(str(OUTPUT_PATH))
    print(f"\n✅ Successfully generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
