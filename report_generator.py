import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Color Palette ─────────────────────────────────────────────────────────────
PRIMARY = colors.HexColor("#2E86AB")
SECONDARY = colors.HexColor("#A23B72")
ACCENT = colors.HexColor("#F18F01")
DARK = colors.HexColor("#1a1a2e")
LIGHT_GRAY = colors.HexColor("#f5f5f5")
MED_GRAY = colors.HexColor("#cccccc")


def generate_pdf_report(profiles, metrics_list, best_algorithm,
                        feature_importance_df, n_customers, feature_names):
    """Generate a professional PDF report of segmentation results."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"],
        fontSize=28, textColor=PRIMARY, spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=13, textColor=SECONDARY, spaceAfter=4,
        alignment=TA_CENTER, fontName="Helvetica"
    )
    h1_style = ParagraphStyle(
        "H1", parent=styles["Heading1"],
        fontSize=16, textColor=PRIMARY, spaceBefore=16,
        spaceAfter=6, fontName="Helvetica-Bold"
    )
    h2_style = ParagraphStyle(
        "H2", parent=styles["Heading2"],
        fontSize=13, textColor=SECONDARY, spaceBefore=10,
        spaceAfter=4, fontName="Helvetica-Bold"
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, spaceAfter=4,
        fontName="Helvetica", leading=14
    )
    small_style = ParagraphStyle(
        "Small", parent=styles["Normal"],
        fontSize=8, textColor=colors.gray,
        fontName="Helvetica", alignment=TA_RIGHT
    )

    story = []

    # ── Cover Page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph("SegmentIQ", title_style))
    story.append(Paragraph("Customer Segmentation Analysis Report", subtitle_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    story.append(Spacer(1, 0.2*inch))

    meta = [
        ["Report Generated:", datetime.now().strftime("%B %d, %Y at %H:%M")],
        ["Total Customers Analysed:", f"{n_customers:,}"],
        ["Best Algorithm:", best_algorithm],
        ["Segments Identified:", str(len(profiles))],
        ["Features Used:", str(len(feature_names))]
    ]
    meta_table = Table(meta, colWidths=[2.5*inch, 3.5*inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph(
        "Built by Lovish Chhabra | linkedin.com/in/lovish-chhabra | github.com/chhabralovish",
        small_style
    ))
    story.append(PageBreak())

    # ── Executive Summary ─────────────────────────────────────────────────────
    story.append(Paragraph("Executive Summary", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=MED_GRAY))
    story.append(Spacer(1, 0.1*inch))

    summary_text = f"""
    This report presents the results of an AI-powered customer segmentation analysis 
    performed on <b>{n_customers:,} customers</b> using multiple clustering algorithms. 
    The analysis identified <b>{len(profiles)} distinct customer segments</b>, each with 
    unique behavioural patterns, value profiles, and engagement characteristics.
    The best performing algorithm was <b>{best_algorithm}</b> based on silhouette score evaluation.
    Each segment has been profiled with tailored business recommendations to drive 
    targeted marketing, loyalty initiatives, and revenue growth strategies.
    """
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 0.2*inch))

    # ── Algorithm Comparison Table ────────────────────────────────────────────
    story.append(Paragraph("Algorithm Performance Comparison", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=MED_GRAY))
    story.append(Spacer(1, 0.1*inch))

    algo_headers = ["Algorithm", "Clusters", "Silhouette ↑", "Davies-Bouldin ↓", "Calinski-Harabasz ↑"]
    algo_data = [algo_headers]
    for m in metrics_list:
        algo_data.append([
            m["algorithm"] + (" ✓" if m["algorithm"] == best_algorithm else ""),
            str(m.get("n_clusters", "N/A")),
            str(m.get("silhouette_score", "N/A")),
            str(m.get("davies_bouldin_score", "N/A")),
            str(m.get("calinski_harabasz_score", "N/A"))
        ])

    algo_table = Table(algo_data, colWidths=[1.8*inch, 1*inch, 1.2*inch, 1.5*inch, 1.5*inch])
    algo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
    ]))
    story.append(algo_table)
    story.append(Spacer(1, 0.2*inch))

    # ── Segment Overview Table ────────────────────────────────────────────────
    story.append(Paragraph("Segment Overview", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=MED_GRAY))
    story.append(Spacer(1, 0.1*inch))

    seg_headers = ["Segment", "Size", "% Share", "Priority Action"]
    seg_data = [seg_headers]
    for p in profiles:
        seg_data.append([
            f"{p['icon']} {p['segment_name']}",
            str(p["size"]),
            f"{p['percentage']}%",
            p["priority"]
        ])

    seg_table = Table(seg_data, colWidths=[2.5*inch, 0.8*inch, 0.9*inch, 2.8*inch])
    seg_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
    ]))
    story.append(seg_table)
    story.append(PageBreak())

    # ── Individual Segment Profiles ───────────────────────────────────────────
    story.append(Paragraph("Detailed Segment Profiles", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=MED_GRAY))

    for p in profiles:
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph(
            f"{p['icon']} Segment {p['cluster_id']}: {p['segment_name']}",
            h2_style
        ))
        story.append(Paragraph(
            f"<b>Size:</b> {p['size']} customers ({p['percentage']}% of total) | "
            f"<b>Priority:</b> {p['priority']}",
            body_style
        ))
        story.append(Paragraph(p["description"], body_style))

        # Key metrics
        means = p["means"]
        key_metrics = []
        for feat in feature_names[:6]:
            if feat in means:
                key_metrics.append([
                    feat.replace("_", " ").title(),
                    f"{means[feat]:.1f}"
                ])

        if key_metrics:
            met_table = Table(key_metrics, colWidths=[3*inch, 2*inch])
            met_table.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, colors.white]),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(met_table)

        # Recommendations
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph("<b>Recommended Actions:</b>", body_style))
        for action in p["actions"]:
            story.append(Paragraph(f"• {action}", body_style))

        story.append(HRFlowable(width="100%", thickness=0.5, color=MED_GRAY))

    story.append(PageBreak())

    # ── Feature Importance ────────────────────────────────────────────────────
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        story.append(Paragraph("Feature Importance Analysis", h1_style))
        story.append(HRFlowable(width="100%", thickness=1, color=MED_GRAY))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            "Features ranked by combined importance score across PCA loadings, "
            "Mutual Information, and Random Forest importance.",
            body_style
        ))
        story.append(Spacer(1, 0.1*inch))

        score_col = "combined_score" if "combined_score" in feature_importance_df.columns else feature_importance_df.columns[-1]
        fi_data = [["Rank", "Feature", "Importance Score"]]
        for i, row in feature_importance_df.head(10).iterrows():
            fi_data.append([
                str(len(fi_data)),
                row["feature"].replace("_", " ").title(),
                f"{row[score_col]:.4f}"
            ])

        fi_table = Table(fi_data, colWidths=[0.8*inch, 3.5*inch, 1.7*inch])
        fi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT_GRAY, colors.white]),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 0), (2, -1), "CENTER"),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, MED_GRAY),
        ]))
        story.append(fi_table)

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Spacer(1, 2*inch))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("SegmentIQ — Customer Segmentation Analysis", subtitle_style))
    story.append(Paragraph(
        "Lovish Chhabra | linkedin.com/in/lovish-chhabra | github.com/chhabralovish",
        subtitle_style
    ))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y')}",
        ParagraphStyle("footer", parent=styles["Normal"],
                       fontSize=9, textColor=colors.gray, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer