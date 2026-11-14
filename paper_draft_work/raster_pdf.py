from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium


def main(pdf_path: Path, out_dir: Path, dpi: int = 150) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72.0
    for index in range(len(doc)):
        page = doc[index]
        bitmap = page.render(scale=scale)
        bitmap.to_pil().convert("RGB").save(out_dir / f"page-{index + 1}.png")
    print(f"pages={len(doc)}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: raster_pdf.py INPUT.pdf OUTPUT_DIR [DPI]")
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3]) if len(sys.argv) == 4 else 150)
