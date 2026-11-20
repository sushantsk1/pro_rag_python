import csv
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .errors import RagError

SUPPORTED_FORMATS = ["pdf", "docx", "pptx", "xlsx", "csv", "txt", "md", "html"]
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=140,
    separators=["\n\n", "\n", " ", ""],
    add_start_index=True,
)


def location(kind="unknown", **values):
    return {"type": kind, "page": None, "slide": None, "sheet": None, "row": None, "heading": None, **values}


def split_block(text: str, loc: dict, *, line_numbers=False) -> list[dict]:
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    chunks = SPLITTER.split_documents([Document(page_content=text, metadata={"location": loc})])
    parts = []
    for chunk in chunks:
        chunk_loc = dict(loc)
        if line_numbers:
            chunk_loc["row"] = text[: chunk.metadata["start_index"]].count("\n") + 1
        parts.append({"text": chunk.page_content, "location": chunk_loc})
    return parts


def table_rows(rows, sheet_name: str, **coordinates) -> list[dict]:
    rows = list(rows)
    if not rows:
        return []
    headers = [str(value).strip() if value is not None else "" for value in rows[0]]
    parts = []
    for row_number, row in enumerate(rows[1:], 2):
        fields = [
            f"{headers[i] if i < len(headers) and headers[i] else f'Column {i + 1}'}: {value}"
            for i, value in enumerate(row)
            if value is not None and str(value).strip()
        ]
        # Very long table rows are split with labels retained as part of the text.
        parts.extend(
            split_block("; ".join(fields), location("sheet", sheet=sheet_name, row=row_number, **coordinates))
        )
    if len(rows) == 1:
        parts.extend(
            split_block("; ".join(headers), location("sheet", sheet=sheet_name, row=1, **coordinates))
        )
    return parts


def markdown_parts(text: str) -> list[dict]:
    parts, block, heading = [], [], None
    preamble = ""
    level = 0
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+", line)
        if match:
            # Carry the title/version preamble into child sections so that
            # policy dates and scope survive structure-aware chunking.
            if level == 1 and len(match[1]) > 1:
                preamble = "\n".join(block).strip()
            parts.extend(
                split_block("\n".join(block), location("heading" if heading else "unknown", heading=heading))
            )
            level = len(match[1])
            if level == 1:
                preamble = ""
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
            block = [preamble, line] if preamble else [line]
        else:
            block.append(line)
    parts.extend(
        split_block("\n".join(block), location("heading" if heading else "unknown", heading=heading))
    )
    return parts


@lru_cache(maxsize=1)
def _ocr_engine():
    from rapidocr import RapidOCR

    return RapidOCR()


def _ocr_page(file_path: Path, page_index: int) -> str:
    try:
        import pypdfium2

        with pypdfium2.PdfDocument(str(file_path)) as pdf:
            bitmap = pdf[page_index].render(scale=2)
            try:
                result = _ocr_engine()(bitmap.to_pil())
                return "\n".join(result.txts or []).strip()
            finally:
                bitmap.close()
    except Exception as cause:
        raise RagError(
            503,
            "OCR_UNAVAILABLE",
            "Local OCR unavailable. Install the ocr extra (RapidOCR, PDFium, and Pillow).",
        ) from cause


def parse_document(file_path: str | Path, file_format: str, ocr=False) -> dict:
    file_path = Path(file_path)
    parts, warnings, ocr_verified = [], [], False
    if file_format not in SUPPORTED_FORMATS:
        raise ValueError("Unsupported document format.")
    if file_format == "txt":
        parts = split_block(file_path.read_text(encoding="utf-8-sig"), location("line"), line_numbers=True)
    elif file_format == "md":
        parts = markdown_parts(file_path.read_text(encoding="utf-8-sig"))
    elif file_format == "html":
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(file_path.read_text(encoding="utf-8-sig"), "html.parser")
        for tag in soup(["script", "style", "head"]):
            tag.decompose()
        lines = []
        for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr", "pre"]):
            prefix = "#" * int(element.name[1]) + " " if re.fullmatch(r"h[1-6]", element.name) else ""
            lines.append(prefix + element.get_text(" ", strip=True))
        parts = (
            markdown_parts("\n".join(lines))
            if lines
            else split_block(soup.get_text("\n", strip=True), location())
        )
    elif file_format == "csv":
        with file_path.open(encoding="utf-8-sig", newline="") as handle:
            parts = table_rows(csv.reader(handle), "Sheet1")
    elif file_format == "xlsx":
        import openpyxl

        workbook = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        try:
            for sheet in workbook:
                parts.extend(table_rows(sheet.iter_rows(values_only=True), sheet.title))
        finally:
            workbook.close()
    elif file_format == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("Encrypted PDFs must be unlocked before upload.")
        for index, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if ocr and not text:
                text = _ocr_page(file_path, index)
                ocr_verified = ocr_verified or bool(text)
            parts.extend(split_block(text, location("page", page=index + 1)))
        if ocr and not ocr_verified:
            warnings.append(
                "OCR found no additional recognized text; only the existing text layer was indexed."
            )
    elif file_format == "docx":
        from docx import Document as WordDocument
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = WordDocument(file_path)
        block, heading = [], None
        for item in doc.iter_inner_content():
            if isinstance(item, Paragraph):
                if item.style and item.style.name.startswith("Heading"):
                    parts.extend(
                        split_block(
                            "\n".join(block), location("heading" if heading else "unknown", heading=heading)
                        )
                    )
                    heading, block = item.text, [item.text]
                else:
                    block.append(item.text)
            elif isinstance(item, Table):
                parts.extend(
                    split_block(
                        "\n".join(block), location("heading" if heading else "unknown", heading=heading)
                    )
                )
                block = []
                parts.extend(
                    table_rows(
                        ([cell.text for cell in row.cells] for row in item.rows), "Table", heading=heading
                    )
                )
        parts.extend(
            split_block("\n".join(block), location("heading" if heading else "unknown", heading=heading))
        )
    elif file_format == "pptx":
        from pptx import Presentation

        for index, slide in enumerate(Presentation(file_path).slides, 1):
            texts = []
            for shape in slide.shapes:
                if shape.has_table:
                    parts.extend(
                        table_rows(
                            ([cell.text for cell in row.cells] for row in shape.table.rows),
                            "Table",
                            slide=index,
                        )
                    )
                elif shape.has_text_frame:
                    texts.append(shape.text)
            parts.extend(split_block("\n".join(texts), location("slide", slide=index)))
    if not parts:
        raise ValueError("No searchable text found. Try local OCR for a scanned PDF.")
    return {
        "parts": parts,
        "mode": "mixed" if ocr_verified else "text",
        "warning": " ".join(warnings) or None,
        "ocrVerified": ocr_verified,
    }


def parse_isolated(file_path: str | Path, file_format: str, ocr=False, timeout=45) -> dict:
    """Keep malformed/slow document parsing out of the API process."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "server.parse_worker", str(file_path), file_format, str(bool(ocr))],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            timeout=120 if ocr else timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        raise RagError(422, "PARSER_TIMEOUT", "Document parsing timed out.") from None
    try:
        parsed = json.loads(result.stdout)
    except (ValueError, UnicodeError):
        raise ValueError("Document parser could not read this file.") from None
    if "error" in parsed:
        problem = parsed["error"]
        raise RagError(422, problem.get("code", "PARSE_ERROR"), problem["message"])
    return parsed
