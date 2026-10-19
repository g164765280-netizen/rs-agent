# -*- coding: utf-8 -*-
"""用 pymupdf 解析全文 PDF：提取正文文本 + 结构化表格（find_tables，无需外部模型）。

输出：corpus/fulltext/{arxiv_id}.json = {title, pages_text:[...], tables:[{page, rows}]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")


def parse_pdf(pdf: Path) -> dict:
    import fitz  # pymupdf
    doc = fitz.open(pdf)
    pages_text = []
    tables = []
    for pno in range(len(doc)):
        page = doc[pno]
        pages_text.append(page.get_text())
        try:
            for t in page.find_tables():
                rows = t.extract()
                if rows:
                    tables.append({"page": pno, "rows": rows})
        except Exception:
            pass
    doc.close()
    return {"pages_text": pages_text, "tables": tables}


def main():
    import json as _j
    meta = [json.loads(l) for l in (CORPUS / "metadata_abstracts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    title_by_id = {p["arxiv_id"]: p.get("title", "") for p in meta}

    pdf_dir = CORPUS / "pdf_fulltext"
    out_dir = CORPUS / "fulltext"
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"parsing {len(pdfs)} PDFs ...", flush=True)
    ok = 0
    for i, pdf in enumerate(pdfs, 1):
        aid = pdf.stem.replace("_", "/")
        out = out_dir / f"{pdf.stem}.json"
        if out.exists():
            ok += 1
            continue
        try:
            parsed = parse_pdf(pdf)
            n_tables = len(parsed["tables"])
            n_chars = sum(len(t) for t in parsed["pages_text"])
            out.write_text(json.dumps({
                "arxiv_id": aid, "title": title_by_id.get(aid, ""),
                **parsed,
            }, ensure_ascii=False), encoding="utf-8")
            ok += 1
            if i % 20 == 0 or i == len(pdfs):
                print(f"[{i}/{len(pdfs)}] {pdf.stem} chars={n_chars} tables={n_tables}", flush=True)
        except Exception as ex:
            print(f"[{i}/{len(pdfs)}] {pdf.stem} FAIL {ex}", flush=True)
    print(f"DONE ok={ok}/{len(pdfs)}", flush=True)


if __name__ == "__main__":
    main()
