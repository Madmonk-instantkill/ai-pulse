"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

# Deterministic (non-agent) service: renders the approved report to PDF and
# emails it via SMTP. Archiving run artifacts is still to be built.

import os
import smtplib
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ai_pulse.report_cleanup import normalize_claim
from ai_pulse.schemas import ExecutiveSummary, GlossaryEntry, PaperSection

BODY_FONT_SIZE = 9
OUTPUT_DIR = Path(__file__).resolve().parents[4] / "outputs"

# First font pair found is used, so Greek letters and symbols in paper text
# (e.g. the eta and nu in a formula) draw correctly. Falls back to the
# built-in Helvetica, which cannot draw them.
FONT_CANDIDATES = [
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]

STATUS_LABELS = {
    "supported": "Verified",
    "partially_supported": "Check source",
}
NOT_CHECKED_LABEL = "Not checked"


def _register_fonts() -> tuple[str, str]:
    """Register the first available TrueType font pair and return its
    (regular, bold) names, or the built-in Helvetica pair if none exist."""
    for regular_path, bold_path in FONT_CANDIDATES:
        if Path(regular_path).exists() and Path(bold_path).exists():
            pdfmetrics.registerFont(TTFont("ReportFont", regular_path))
            pdfmetrics.registerFont(TTFont("ReportFont-Bold", bold_path))
            # Links the pair so <b> and <i> inside Paragraphs work; there
            # is no italic file, so italics fall back to the regular font.
            pdfmetrics.registerFontFamily(
                "ReportFont", normal="ReportFont", bold="ReportFont-Bold",
                italic="ReportFont", boldItalic="ReportFont-Bold",
            )
            return "ReportFont", "ReportFont-Bold"
    return "Helvetica", "Helvetica-Bold"


def _build_styles(regular: str, bold: str) -> dict[str, ParagraphStyle]:
    """All paragraph styles used by the report, at one shared font size."""
    body = ParagraphStyle(
        "body", fontName=regular, fontSize=BODY_FONT_SIZE,
        leading=BODY_FONT_SIZE * 1.4, spaceAfter=4,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "title", parent=body, fontName=bold, fontSize=18, leading=22,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "meta", parent=body, textColor=colors.grey, spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "h1", parent=body, fontName=bold, fontSize=13, leading=16,
            spaceBefore=6, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "h2", parent=body, fontName=bold, fontSize=10, leading=13,
            spaceBefore=6, spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=body, leftIndent=12, bulletIndent=2,
            spaceAfter=2,
        ),
        "cell": ParagraphStyle(
            "cell", parent=body, fontSize=BODY_FONT_SIZE - 1,
            leading=(BODY_FONT_SIZE - 1) * 1.35, spaceAfter=0,
        ),
    }


def _text(value: str) -> str:
    """Escape text for reportlab Paragraphs, which read <, > and & as
    markup (claims like "e0 < 1/2" would otherwise break the PDF)."""
    return escape(value)


def _bullets(items: list[str], styles: dict) -> list:
    """One bulleted Paragraph per item."""
    return [
        Paragraph(_text(item), styles["bullet"], bulletText="\u2022")
        for item in items
    ]


def _paper_pages(
    section: PaperSection, title: str, styles: dict
) -> list:
    """The body pages for one paper: title, link, the three explanation
    fields, key results and limitations. Empty lists are skipped."""
    story = [
        Paragraph(_text(title), styles["h1"]),
        Paragraph(
            f'arXiv: <link href="https://arxiv.org/abs/{section.arxiv_id}" '
            f'color="blue">{section.arxiv_id}</link>',
            styles["meta"],
        ),
        Paragraph("What it is about", styles["h2"]),
        Paragraph(_text(section.what_its_about), styles["body"]),
        Paragraph("What is new", styles["h2"]),
        Paragraph(_text(section.whats_new), styles["body"]),
        Paragraph("Why it matters", styles["h2"]),
        Paragraph(_text(section.why_it_matters), styles["body"]),
    ]
    if section.key_results:
        story.append(Paragraph("Key results", styles["h2"]))
        story += _bullets(section.key_results, styles)
    if section.limitations:
        story.append(Paragraph("Limitations", styles["h2"]))
        story += _bullets(section.limitations, styles)
    return story


def _ledger_table(
    section: PaperSection, claim_statuses: dict[str, str], styles: dict
) -> Table | None:
    """The source ledger for one paper: claim, pages, verification status.
    Returns None if the paper has no claims left."""
    if not section.claim_ledger:
        return None
    rows = [[
        Paragraph("Claim", styles["cell"]),
        Paragraph("Source pages", styles["cell"]),
        Paragraph("Status", styles["cell"]),
    ]]
    for claim in section.claim_ledger:
        status = claim_statuses.get(normalize_claim(claim.claim_text), "")
        rows.append([
            Paragraph(_text(claim.claim_text), styles["cell"]),
            Paragraph(_text(", ".join(claim.source_pages)), styles["cell"]),
            Paragraph(
                STATUS_LABELS.get(status, NOT_CHECKED_LABEL), styles["cell"]
            ),
        ])
    table = Table(rows, colWidths=[10.4 * cm, 3.6 * cm, 2.6 * cm],
                  repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return table


def _glossary_pages(
    glossary: dict[str, list[GlossaryEntry]],
    titles: dict[str, str],
    styles: dict,
) -> list:
    """The glossary section: entries grouped under each paper. Empty if
    no paper has entries."""
    story = []
    for arxiv_id, entries in glossary.items():
        if not entries:
            continue
        story.append(Paragraph(
            _text(titles.get(arxiv_id, arxiv_id)), styles["h2"]
        ))
        for entry in entries:
            pages = ", ".join(entry.source_pages)
            story.append(Paragraph(
                f"<b>{_text(entry.term)}</b>: {_text(entry.plain_definition)}"
                f" <i>Why it matters:</i> {_text(entry.why_it_matters)}"
                f" <i>Example:</i> {_text(entry.example)}"
                + (f" <i>(source: {_text(pages)})</i>" if pages else ""),
                styles["body"],
            ))
    if story:
        story.insert(0, Paragraph("Glossary", styles["h1"]))
    return story


def _draw_page_number(canvas, doc) -> None:
    """Footer: page number, bottom centre."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf_report(
    summary: ExecutiveSummary,
    sections: list[PaperSection],
    titles: dict[str, str],
    claim_statuses: dict[str, dict[str, str]],
    glossary: dict[str, list[GlossaryEntry]] | None = None,
    report_date: date | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Write the final report PDF and return its path.

    sections should already be cleaned (rejected claims removed) by
    report_cleanup.remove_rejected_claims; claim_statuses maps each
    arxiv_id to that function's status dict. titles maps arxiv_id to the
    paper title (falls back to the arXiv ID). glossary is optional, and
    the Glossary section is left out when it is empty. The date is
    supplied by code, never by the model.
    """
    report_date = report_date or date.today()
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"AI_Pulse_{report_date.isoformat()}.pdf"

    regular, bold = _register_fonts()
    styles = _build_styles(regular, bold)

    story = [
        Paragraph(_text(summary.report_title), styles["title"]),
        Paragraph(report_date.strftime("%d %B %Y"), styles["meta"]),
        Paragraph("Executive summary", styles["h1"]),
        Paragraph(_text(summary.summary), styles["body"]),
        Paragraph("In this issue", styles["h2"]),
    ]
    story += _bullets(
        [titles.get(s.arxiv_id, s.arxiv_id) for s in sections], styles
    )

    for section in sections:
        story.append(PageBreak())
        story += _paper_pages(
            section, titles.get(section.arxiv_id, section.arxiv_id), styles
        )

    story += [PageBreak(), Paragraph("Source notes", styles["h1"]),
              Paragraph(
                  "Every claim below was checked against the pages of the "
                  "paper it cites. <b>Verified</b>: the pages state it. "
                  "<b>Check source</b>: the pages state only part of it or "
                  "with a different number, so read the source. Claims the "
                  "pages did not support were removed.",
                  styles["body"])]
    for section in sections:
        table = _ledger_table(
            section, claim_statuses.get(section.arxiv_id, {}), styles
        )
        if table is None:
            continue
        story.append(Paragraph(
            _text(titles.get(section.arxiv_id, section.arxiv_id)),
            styles["h2"],
        ))
        story.append(table)
        story.append(Spacer(1, 6))

    glossary_story = _glossary_pages(glossary or {}, titles, styles)
    if glossary_story:
        story.append(PageBreak())
        story += glossary_story

    SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=summary.report_title, author="AI Pulse",
    ).build(story, onFirstPage=_draw_page_number,
            onLaterPages=_draw_page_number)
    return output_path


def send_report_email(pdf_path: Path, subject: str, body: str) -> None:
    """Email the PDF as an attachment over SMTP with STARTTLS.

    Settings come from the environment (.env): SMTP_HOST, SMTP_PORT
    (defaults to 587), SMTP_USER, SMTP_PASSWORD and EMAIL_TO. Gmail needs
    an app password here, not the normal account password. Raises
    RuntimeError naming any missing setting (never printing values), and
    lets smtplib's own errors through if the login or send fails."""
    settings = {
        name: os.environ.get(name, "").strip()
        for name in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "EMAIL_TO")
    }
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise RuntimeError(
            "Email settings missing in .env: " + ", ".join(missing)
        )
    port = int(os.environ.get("SMTP_PORT", "").strip() or 587)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings["SMTP_USER"]
    message["To"] = settings["EMAIL_TO"]
    message.set_content(body)
    message.add_attachment(
        pdf_path.read_bytes(), maintype="application", subtype="pdf",
        filename=pdf_path.name,
    )

    with smtplib.SMTP(settings["SMTP_HOST"], port, timeout=60) as server:
        server.starttls()
        server.login(settings["SMTP_USER"], settings["SMTP_PASSWORD"])
        server.send_message(message)
