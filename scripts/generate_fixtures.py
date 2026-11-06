"""Regenerate synthetic parser fixtures using Python libraries (no model/API calls)."""

from pathlib import Path


def main():
    from docx import Document
    from openpyxl import Workbook
    from PIL import Image, ImageDraw, ImageFont
    from pptx import Presentation
    from pptx.util import Inches
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    directory = Path(__file__).resolve().parents[1] / "tests" / "integration" / "fixtures"
    directory.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(directory / "shipping.pdf"), invariant=1)
    pdf.drawString(72, 720, "Harbor shipping is free on orders above 50 dollars.")
    pdf.save()
    image = Image.new("RGB", (1000, 180), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 50), "SCANNED PROOF", fill="black", font=ImageFont.load_default(size=70))
    pdf = canvas.Canvas(str(directory / "scanned.pdf"), invariant=1)
    pdf.drawImage(ImageReader(image), 36, 600, width=540, height=100)
    pdf.save()
    doc = Document()
    doc.add_paragraph("The Apollo return window is 42 days after delivery.")
    doc.save(directory / "returns.docx")
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    textbox = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(8), Inches(2))
    textbox.text_frame.text = "Project Nova launches in July 2026."
    presentation.save(directory / "launch.pptx")
    workbook = Workbook()
    workbook.active.title = "Metrics"
    workbook.active.append(["Metric", "Value"])
    workbook.active.append(["Revenue", 120000])
    workbook.save(directory / "metrics.xlsx")
    texts = {
        "products.csv": "product,region,inventory\nBeacon,West,73\nOrbit,East,28\n",
        "policy.txt": "Aster warranty coverage lasts 24 months from the date of purchase.\n",
        "onboarding.md": "# Onboarding\nThe Helix onboarding owner is Priya Shah.\n",
        "incident.html": "<!doctype html><html><body><h1>Incident response</h1><p>The first escalation step is to page the on-call engineer.</p></body></html>",
        "conflict-a.txt": "The Helix support desk closes at 18:00 on weekdays.\n",
        "conflict-b.txt": "The Helix support desk closes at 20:00 on weekdays.\n",
        "incremental.txt": "The Lyra retention policy keeps records for 90 days.\n",
        "empty.txt": "",
        "broken.pdf": "%PDF-1.4\nnot a real PDF",
    }
    for name, content in texts.items():
        (directory / name).write_text(content, encoding="utf-8")
    (directory / "unsupported.bin").write_bytes(bytes(range(5)))
    print(f"Generated fixtures in {directory}")


if __name__ == "__main__":
    main()
