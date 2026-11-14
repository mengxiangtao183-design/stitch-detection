from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def paragraph_payload(paragraph):
    return {
        "style": paragraph.style.name if paragraph.style else "",
        "text": clean(paragraph.text),
        "alignment": str(paragraph.alignment),
    }


def extract(path: Path, out_dir: Path) -> None:
    doc = Document(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    body = doc.element.body
    p_index = 0
    t_index = 0
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            p = doc.paragraphs[p_index]
            p_index += 1
            payload = paragraph_payload(p)
            payload.update({"kind": "paragraph", "index": p_index})
            if payload["text"]:
                records.append(payload)
        elif child.tag == qn("w:tbl"):
            table = doc.tables[t_index]
            t_index += 1
            rows = []
            for row in table.rows:
                rows.append([clean(" ".join(p.text for p in cell.paragraphs)) for cell in row.cells])
            records.append({"kind": "table", "index": t_index, "rows": rows})

    headers = []
    footers = []
    for s_index, section in enumerate(doc.sections, start=1):
        headers.append(
            {
                "section": s_index,
                "paragraphs": [paragraph_payload(p) for p in section.header.paragraphs if clean(p.text)],
                "tables": [
                    [[clean(" ".join(p.text for p in c.paragraphs)) for c in r.cells] for r in t.rows]
                    for t in section.header.tables
                ],
            }
        )
        footers.append(
            {
                "section": s_index,
                "paragraphs": [paragraph_payload(p) for p in section.footer.paragraphs if clean(p.text)],
                "tables": [
                    [[clean(" ".join(p.text for p in c.paragraphs)) for c in r.cells] for r in t.rows]
                    for t in section.footer.tables
                ],
            }
        )

    with zipfile.ZipFile(path) as zf:
        parts = []
        all_text = {}
        media = []
        for info in zf.infolist():
            parts.append({"name": info.filename, "size": info.file_size})
            if info.filename.startswith("word/") and info.filename.endswith(".xml"):
                raw = zf.read(info.filename).decode("utf-8", errors="replace")
                texts = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", raw, flags=re.S)
                if texts:
                    stripped = [clean(re.sub(r"<[^>]+>", "", t)) for t in texts]
                    all_text[info.filename] = [t for t in stripped if t]
            if info.filename.startswith("word/media/") and not info.is_dir():
                target = out_dir / "media" / Path(info.filename).name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(zf.read(info.filename))
                media.append(str(target))

    payload = {
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "sections": len(doc.sections),
        "paragraph_count": len(doc.paragraphs),
        "table_count": len(doc.tables),
        "inline_shape_count": len(doc.inline_shapes),
        "records": records,
        "headers": headers,
        "footers": footers,
        "package_parts": parts,
        "all_part_text": all_text,
        "media": media,
    }
    (out_dir / "extracted.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# {path.name}", ""]
    for rec in records:
        if rec["kind"] == "paragraph":
            lines.append(f"[{rec['style']}] {rec['text']}")
        else:
            lines.append(f"\n[TABLE {rec['index']}]")
            for row in rec["rows"]:
                lines.append(" | ".join(row))
            lines.append("")
    (out_dir / "content.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_docx.py INPUT.docx OUTPUT_DIR")
    extract(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
