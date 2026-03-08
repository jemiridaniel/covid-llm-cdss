"""
PDF clinical report generation using ReportLab.
COVID-19 CDSS — includes SHAP top features section.
"""
from datetime import datetime
from io import BytesIO
from typing import Any, Dict, List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Color constants ─────────────────────────────────────────────────────────────
PRIMARY      = colors.HexColor("#0284C7")
LIGHT_BG     = colors.HexColor("#F0F9FF")
BORDER_COLOR = colors.HexColor("#E2E8F0")
TEXT         = colors.HexColor("#0F172A")
TEXT_MUTED   = colors.HexColor("#64748B")

STAGE_COLORS = {
    "Critical":  colors.HexColor("#7F1D1D"),
    "Severe":    colors.HexColor("#DC2626"),
    "Moderate":  colors.HexColor("#EA580C"),
    "Mild":      colors.HexColor("#D97706"),
    "No_COVID":  colors.HexColor("#16A34A"),
}
STAGE_LABELS = {
    "Critical":  "CRITICAL — Emergency Care Required",
    "Severe":    "Severe COVID-19",
    "Moderate":  "Moderate COVID-19",
    "Mild":      "Mild COVID-19",
    "No_COVID":  "No COVID-19 Detected",
}


def _section_label(text: str) -> Paragraph:
    return Paragraph(
        text,
        ParagraphStyle(
            "SL", fontSize=8, fontName="Helvetica-Bold",
            textColor=TEXT_MUTED, spaceBefore=12, spaceAfter=5,
        ),
    )


def _body(text: str, bold: bool = False, text_color: Any = None) -> Paragraph:
    return Paragraph(
        text,
        ParagraphStyle(
            "Body", fontSize=10,
            fontName="Helvetica-Bold" if bold else "Helvetica",
            textColor=text_color or TEXT, leading=15,
        ),
    )


def _info_table(rows: List[List[str]]) -> Table:
    col_widths = [3.5 * cm, 7.5 * cm, 3 * cm, 3.5 * cm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 0), (0, -1),  TEXT_MUTED),
        ("TEXTCOLOR",     (2, 0), (2, -1),  TEXT_MUTED),
        ("TEXTCOLOR",     (1, 0), (-1, -1), TEXT),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS",(0, 0), (-1, -1), [LIGHT_BG, colors.white]),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("INNERGRID",     (0, 0), (-1, -1), 0.5, BORDER_COLOR),
    ]))
    return t


def generate_report(data: Dict[str, Any]) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2.5 * cm,
    )

    story = []

    # ── Header banner ───────────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "HTitle", fontSize=15, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER, leading=20,
    )
    sub_style = ParagraphStyle(
        "HSub", fontSize=9, fontName="Helvetica",
        textColor=colors.HexColor("#BAE6FD"), alignment=TA_CENTER, leading=14,
    )
    banner = Table(
        [
            [Paragraph("COVID-19 Clinical Decision Support Report", title_style)],
            [Paragraph("ML Severity Assessment  |  SHAP Explainability  |  For Clinical Decision Support Only", sub_style)],
        ],
        colWidths=[doc.width],
    )
    banner.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), PRIMARY),
        ("TOPPADDING",    (0, 0), (-1, 0),  14),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 16),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 16),
    ]))
    story.append(banner)
    story.append(Spacer(1, 0.5 * cm))

    # ── Patient Information ─────────────────────────────────────────────────────
    story.append(_section_label("PATIENT INFORMATION"))
    patient_name = data.get("patient_name") or "Not provided"
    patient_id   = data.get("patient_id")   or "Not provided"
    timestamp    = data.get("timestamp")    or datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(_info_table([
        ["Patient Name:", patient_name, "Patient ID:", patient_id],
        ["Report Date/Time:", timestamp, "", ""],
    ]))

    # ── Demographics & Clinical Status ──────────────────────────────────────────
    story.append(_section_label("DEMOGRAPHICS & CLINICAL STATUS"))

    def yn(val) -> str:
        if isinstance(val, bool):
            return "Yes" if val else "No"
        return str(val).capitalize() if val else "No"

    age       = str(data.get("age", "Unknown"))
    sex       = str(data.get("sex", "Unknown")).capitalize()
    pregnant  = yn(data.get("pregnant", False))
    hosp      = yn(data.get("hospitalized", False))
    pneumonia = yn(data.get("pneumonia", False))

    story.append(_info_table([
        ["Age:",        age,      "Sex:",              sex],
        ["Pregnant:",   pregnant, "Hospitalized:",     hosp],
        ["Pneumonia:",  pneumonia, "",                 ""],
    ]))

    # ── Comorbidities ───────────────────────────────────────────────────────────
    story.append(_section_label("COMORBIDITIES"))
    comorbidities = [
        ("Diabetes",            data.get("diabetes", False)),
        ("COPD",                data.get("copd", False)),
        ("Asthma",              data.get("asthma", False)),
        ("Immunocompromised",   data.get("immunocompromised", False)),
        ("Hypertension",        data.get("hypertension", False)),
        ("Cardiovascular",      data.get("cardiovascular", False)),
        ("Obesity",             data.get("obesity", False)),
        ("Chronic Renal",       data.get("renal_chronic", False)),
        ("Tobacco use",         data.get("tobacco", False)),
    ]
    active = [name for name, val in comorbidities if val]
    if active:
        rows = []
        for i in range(0, len(active), 3):
            chunk = active[i : i + 3]
            while len(chunk) < 3:
                chunk.append("")
            rows.append([f"• {s}" if s else "" for s in chunk])
        comor_t = Table(rows, colWidths=[doc.width / 3] * 3)
        comor_t.setStyle(TableStyle([
            ("FONTSIZE",       (0, 0), (-1, -1), 9.5),
            ("FONTNAME",       (0, 0), (-1, -1), "Helvetica"),
            ("TEXTCOLOR",      (0, 0), (-1, -1), TEXT),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_BG, colors.white]),
            ("BOX",            (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID",      (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ]))
        story.append(comor_t)
    else:
        story.append(_body("No comorbidities reported."))

    # ── Severity Assessment ─────────────────────────────────────────────────────
    story.append(_section_label("ML SEVERITY ASSESSMENT"))
    severity    = data.get("severity", "Unknown")
    stage_color = STAGE_COLORS.get(severity, PRIMARY)
    stage_label = STAGE_LABELS.get(severity, severity)
    confidence  = data.get("confidence", 0.0)

    badge_style = ParagraphStyle(
        "Badge", fontSize=12, fontName="Helvetica-Bold",
        textColor=colors.white, alignment=TA_CENTER, leading=16,
    )
    conf_style = ParagraphStyle(
        "Conf", fontSize=10, fontName="Helvetica",
        textColor=colors.white, alignment=TA_CENTER, leading=16,
    )
    diag_row = Table(
        [[Paragraph(stage_label, badge_style),
          Paragraph(f"Confidence: {confidence*100:.1f}%", conf_style)]],
        colWidths=[doc.width * 0.65, doc.width * 0.35],
    )
    diag_row.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), stage_color),
        ("BACKGROUND",    (1, 0), (1, 0), colors.HexColor("#1E293B")),
        ("TOPPADDING",    (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 11),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
    ]))
    story.append(diag_row)

    # ── SHAP Top Features ───────────────────────────────────────────────────────
    story.append(_section_label("TOP CONTRIBUTING FACTORS (SHAP)"))
    shap_features = data.get("shap_features", [])
    if shap_features:
        shap_rows = [["Feature", "SHAP Value", "Direction"]]
        for feat in shap_features:
            if isinstance(feat, dict):
                fname  = feat.get("feature", "")
                shap_v = feat.get("shap_value", 0.0)
            else:
                fname  = feat.feature
                shap_v = feat.shap_value
            direction = "↑ Increases severity" if shap_v > 0 else "↓ Decreases severity"
            shap_rows.append([fname, f"{shap_v:+.4f}", direction])

        shap_t = Table(shap_rows, colWidths=[doc.width * 0.45, doc.width * 0.2, doc.width * 0.35])
        shap_t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [LIGHT_BG, colors.white]),
            ("BOX",           (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("INNERGRID",     (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ]))
        story.append(shap_t)
    else:
        story.append(_body("SHAP features not available."))

    # ── AI Clinical Explanation ─────────────────────────────────────────────────
    story.append(_section_label("AI CLINICAL EXPLANATION"))
    reason_t = Table(
        [[_body(data.get("llm_explanation") or "No explanation available.")]],
        colWidths=[doc.width],
    )
    reason_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), LIGHT_BG),
        ("BOX",           (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    story.append(reason_t)
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(
        f"Model: {data.get('model_used', 'unknown')}",
        ParagraphStyle("Mi", fontSize=8, fontName="Helvetica-Oblique", textColor=TEXT_MUTED),
    ))

    # ── Treatment Recommendation ────────────────────────────────────────────────
    story.append(_section_label("TREATMENT RECOMMENDATION"))
    is_critical  = severity in ("Critical", "Severe")
    presc_bg     = colors.HexColor("#FEF2F2") if is_critical else LIGHT_BG
    presc_border = colors.HexColor("#FECACA") if is_critical else BORDER_COLOR
    presc_color  = colors.HexColor("#991B1B") if is_critical else TEXT
    presc_t = Table(
        [[_body(data.get("treatment_recommendation", ""), bold=is_critical, text_color=presc_color)]],
        colWidths=[doc.width],
    )
    presc_t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), presc_bg),
        ("BOX",           (0, 0), (-1, -1), 1.0, presc_border),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
    ]))
    story.append(presc_t)

    # ── Footer disclaimer ───────────────────────────────────────────────────────
    story.append(Spacer(1, 0.7 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_COLOR))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by an AI-powered clinical decision support tool. "
        "Final diagnosis must be confirmed by a licensed clinician. "
        "This document does not constitute medical advice and should not replace professional medical consultation. "
        "In a medical emergency, call your local emergency services immediately.",
        ParagraphStyle(
            "Disc", fontSize=7.5, fontName="Helvetica-Oblique",
            textColor=TEXT_MUTED, alignment=TA_CENTER, leading=11,
        ),
    ))

    doc.build(story)
    return buffer.getvalue()
