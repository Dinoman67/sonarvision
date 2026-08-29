import os
import io
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

def create_pdf_report(
    analysis_data: Dict[str, Any],
    annotated_image_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> bytes:
    """
    Builds a professional, comprehensive PDF analysis report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        output_path if output_path else buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    header_title_style = ParagraphStyle(
        'HeaderTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f172a')
    )
    
    header_subtitle_style = ParagraphStyle(
        'HeaderSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#64748b')
    )

    section_heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0369a1'),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1e293b')
    )

    badge_yes_style = ParagraphStyle(
        'BadgeYes',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor('#059669'),
        alignment=TA_CENTER
    )

    badge_no_style = ParagraphStyle(
        'BadgeNo',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_CENTER
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>YOLO-ESI // DEBRIS INTELLIGENCE REPORT</b>", header_title_style),
            Paragraph(f"<b>Generated:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}<br/><b>Analysis ID:</b> {analysis_data.get('analysis_id', 'N/A')[:12]}", header_subtitle_style)
        ]
    ]
    t_header = Table(header_data, colWidths=[340, 200])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=8, spaceBefore=4))

    # 2. Executive Summary Callout Box
    summary = analysis_data.get("summary", {})
    file_meta = analysis_data.get("file_metadata", {})
    geo_meta = analysis_data.get("geospatial_metadata", {})
    model_meta = analysis_data.get("model_metadata", {})
    detections = analysis_data.get("detections", [])

    debris_detected = summary.get("debris_detected", False)
    total_dets = summary.get("total_detections", 0)
    max_conf = summary.get("highest_confidence")
    avg_conf = summary.get("average_confidence")
    inf_time = summary.get("inference_time_ms", 0.0)

    summary_table_data = [
        [
            Paragraph("<b>STATUS:</b>", body_style),
            Paragraph(f"<b>{'DEBRIS DETECTED' if debris_detected else 'NO DEBRIS DETECTED'}</b>", badge_yes_style if debris_detected else badge_no_style),
            Paragraph("<b>TOTAL DETECTIONS:</b>", body_style),
            Paragraph(f"<b>{total_dets}</b>", body_style),
            Paragraph("<b>MAX CONFIDENCE:</b>", body_style),
            Paragraph(f"<b>{max_conf*100:.1f}%</b>" if max_conf else "N/A", body_style),
        ],
        [
            Paragraph("<b>IMAGE FILE:</b>", body_style),
            Paragraph(f"{file_meta.get('filename', 'N/A')}", body_style),
            Paragraph("<b>IMAGE DIMENSIONS:</b>", body_style),
            Paragraph(f"{file_meta.get('width', 0)} × {file_meta.get('height', 0)} px", body_style),
            Paragraph("<b>INFERENCE LATENCY:</b>", body_style),
            Paragraph(f"{inf_time:.1f} ms", body_style),
        ]
    ]
    t_summary = Table(summary_table_data, colWidths=[90, 90, 95, 85, 90, 90])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 8))

    # 3. Model & Geospatial Metadata
    story.append(Paragraph("1. System & Geospatial Metadata", section_heading_style))
    meta_table_data = [
        [
            Paragraph("<b>Model Name:</b>", body_style), Paragraph(str(model_meta.get("model_name")), body_style),
            Paragraph("<b>Georeferenced:</b>", body_style), Paragraph(f"<b>{'CAMERA GPS' if geo_meta.get('camera_latitude') is not None else ('YES' if geo_meta.get('georeferenced') else 'NO')}</b>", body_style),
        ],
        [
            Paragraph("<b>Architecture:</b>", body_style), Paragraph(str(model_meta.get("architecture")), body_style),
            Paragraph("<b>Coordinate System:</b>", body_style), Paragraph(str(geo_meta.get("crs") or "Unavailable"), body_style),
        ],
        [
            Paragraph("<b>Model Format:</b>", body_style), Paragraph(f"{model_meta.get('format')} ({model_meta.get('input_resolution')})", body_style),
            Paragraph("<b>Coordinate Source:</b>", body_style), Paragraph(str(geo_meta.get("coordinate_source") or "Unavailable"), body_style),
        ],
        [
            Paragraph("<b>Execution Provider:</b>", body_style), Paragraph(str(model_meta.get("execution_provider")), body_style),
            Paragraph("<b>Pixel Resolution:</b>", body_style), Paragraph(f"{geo_meta.get('pixel_resolution')[0]} × {geo_meta.get('pixel_resolution')[1]} m/px" if geo_meta.get('pixel_resolution') else "N/A", body_style),
        ],
        [
            Paragraph("<b>Model SHA-256:</b>", body_style), Paragraph(f"<font size='6.5'>{model_meta.get('sha256_hash', 'N/A')[:32]}...</font>", body_style),
            Paragraph("<b>Geotag Status:</b>", body_style), Paragraph(f"Camera: {geo_meta.get('camera_latitude'):.6f}°, {geo_meta.get('camera_longitude'):.6f}°" if geo_meta.get("camera_latitude") is not None else str(geo_meta.get("status_message")), body_style),
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[95, 175, 95, 175])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
        ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # 4. Embedded Annotated Image
    if annotated_image_path and os.path.exists(annotated_image_path):
        story.append(Paragraph("2. Annotated Detection Imagery", section_heading_style))
        try:
            img = RLImage(annotated_image_path, width=540, height=220)
            story.append(img)
            story.append(Spacer(1, 6))
        except Exception:
            pass

    # 5. Detection Summary Table
    story.append(Paragraph("3. Target Detection Inventory", section_heading_style))
    
    det_headers = ["ID", "Class", "Confidence", "Pixel Bounds [X1, Y1, X2, Y2]", "Center (X, Y)", "Latitude", "Longitude"]
    det_table_data = [[Paragraph(f"<b>{h}</b>", body_style) for h in det_headers]]

    if len(detections) == 0:
        det_table_data.append([
            Paragraph("—", body_style),
            Paragraph("No objects detected above threshold", body_style),
            Paragraph("—", body_style),
            Paragraph("—", body_style),
            Paragraph("—", body_style),
            Paragraph("—", body_style),
            Paragraph("—", body_style),
        ])
    else:
        for det in detections[:30]:  # Cap to top 30 in PDF
            box = det.get("bbox", {})
            cp = det.get("center_pixel", {})
            geo = det.get("geolocation") or {}
            
            box_str = f"[{box.get('x1', 0):.0f}, {box.get('y1', 0):.0f}, {box.get('x2', 0):.0f}, {box.get('y2', 0):.0f}]"
            center_str = f"({cp.get('x', 0):.0f}, {cp.get('y', 0):.0f})"
            lat_str = f"{geo.get('latitude'):.6f}°" if geo.get('latitude') is not None else "N/A"
            lon_str = f"{geo.get('longitude'):.6f}°" if geo.get('longitude') is not None else "N/A"

            det_table_data.append([
                Paragraph(f"#{det.get('id'):02d}", body_style),
                Paragraph(f"{det.get('class_name')}", body_style),
                Paragraph(f"<b>{det.get('confidence', 0)*100:.1f}%</b>", body_style),
                Paragraph(box_str, body_style),
                Paragraph(center_str, body_style),
                Paragraph(lat_str, body_style),
                Paragraph(lon_str, body_style),
            ])

    t_dets = Table(det_table_data, colWidths=[30, 80, 60, 140, 80, 75, 75])
    t_dets.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0284c7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_dets)
    story.append(Spacer(1, 8))

    # 6. Technical Interpretation
    story.append(Paragraph("4. Technical Interpretation & Assessment", section_heading_style))
    interp_text = (
        f"The YOLOv8-ESI detector evaluated the input scene '{file_meta.get('filename')}' using spatial-aware "
        f"Squeeze-and-Excitation channel attention. A total of <b>{total_dets}</b> candidate marine debris objects "
        f"exceeded the operational confidence threshold. "
    )
    if geo_meta.get("georeferenced"):
        interp_text += (
            f"Geographic coordinates were successfully mapped to the target scene using the genuine {geo_meta.get('crs')} "
            f"affine transformation matrix. Target locations can be referenced directly on hydrographic and GIS basemaps."
        )
    else:
        interp_text += (
            "The analyzed imagery did not contain valid georeferencing tags or affine transforms; "
            "pixel locations are recorded accurately, while geospatial coordinates remain unpopulated per protocol."
        )

    disclaimer_text = (
        "<font size='7' color='#64748b'><b>Operational Disclaimer:</b> This report is generated automatically by the "
        "YOLOv8-ESI remote sensing inference pipeline. The model functions as a visual and acoustic signature detector. "
        "Detections indicate acoustic/visual anomaly targets consistent with marine debris signatures and do not certify "
        "hazardous material composition.</font>"
    )

    story.append(Paragraph(interp_text, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(disclaimer_text, body_style))

    # Build PDF
    doc.build(story)

    if output_path:
        with open(output_path, "rb") as f:
            return f.read()
    return buffer.getvalue()
