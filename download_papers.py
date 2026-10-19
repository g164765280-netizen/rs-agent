# -*- coding: utf-8 -*-
"""从 arXiv 下载论文 PDF 与摘要（纯 stdlib，无第三方依赖）。

用法：
  python3 download_papers.py --manifest papers_manifest.json --out /path/to/corpus

流程：
  1. 读 manifest（title/aka/authors/year/category）
  2. 对每篇：arXiv API 按标题检索 → 选最相似 → 得到 arxiv_id/abstract/pdf 链接
  3. 下载 PDF，保存 metadata.jsonl（含 arxiv_id/标题/摘要/类别/年份）
  4. 输出：corpus/pdf/*.pdf + corpus/metadata.jsonl + corpus/resolve_report.json
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "rscd-paper-corpus/1.0 (research assistant)"}
ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def fetch(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        eid = e.findtext("a:id", default="", namespaces=NS)
        title = re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=NS)).strip()
        summary = re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=NS)).strip()
        arxiv_id = eid.split("/abs/")[-1] if "/abs/" in eid else ""
        out.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": summary,
            "pdf": eid.replace("/abs/", "/pdf/") + ".pdf" if arxiv_id else "",
        })
    return out


def search_arxiv(title: str, aka: str = "") -> list[dict]:
    candidates = []
    queries = []
    if title:
        queries.append(f'ti:"{title}"')
    if aka:
        queries.append(f'all:"{aka}" remote sensing change detection"')
    # 兜底：取标题关键词
    words = norm(title).split()[:6]
    if len(words) >= 3:
        queries.append(f'all:{" ".join(words)}')
    for q in queries:
        if candidates:
            break
        url = ARXIV_API + "?" + urllib.parse.urlencode({"search_query": q, "max_results": 5})
        try:
            data = fetch(url).decode("utf-8")
            entries = parse_entries(data)
            if entries:
                candidates = entries
        except Exception as ex:
            print(f"  query failed: {q[:60]}... -> {ex}", file=sys.stderr)
        time.sleep(3)
    return candidates


def best_match(entries: list[dict], title: str) -> dict | None:
    if not entries:
        return None
    nt = norm(title)
    scored = []
    for e in entries:
        s = difflib.SequenceMatcher(None, nt, norm(e["title"])).ratio()
        # 标题包含关系加权
        if nt and (nt in norm(e["title"]) or norm(e["title"]) in nt):
            s += 0.3
        scored.append((s, e))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    papers = manifest["papers"]
    out = Path(args.out)
    pdf_dir = out / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    meta_lines = []
    report = {"total": len(papers), "resolved": 0, "downloaded": 0, "failed": []}

    for i, p in enumerate(papers, 1):
        title = p["title"]
        aka = p.get("aka", "")
        print(f"[{i}/{len(papers)}] {p.get('category','')} | {aka or ''} | {title[:70]}")
        cands = search_arxiv(title, aka)
        hit = best_match(cands, title)
        if not hit:
            print(f"  -> NOT RESOLVED")
            report["failed"].append({"title": title, "reason": "no_match"})
            continue
        report["resolved"] += 1
        arxiv_id = hit["arxiv_id"]
        rec = {
            "title": p["title"],
            "aka": aka,
            "category": p.get("category", ""),
            "year": p.get("year"),
            "authors": p.get("authors", ""),
            "arxiv_id": arxiv_id,
            "arxiv_title": hit["title"],
            "abstract": hit["abstract"],
        }
        # 下载 PDF
        pdf_path = pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"
        try:
            if not pdf_path.exists():
                data = fetch(hit["pdf"], timeout=90)
                pdf_path.write_bytes(data)
            size = pdf_path.stat().st_size
            ok = size > 10_000 and data[:4] == b"%PDF" if pdf_path.exists() else False
            rec["pdf_bytes"] = size
            rec["pdf_ok"] = ok
            report["downloaded"] += (1 if ok else 0)
            print(f"  -> {arxiv_id}  {size//1024}KB  pdf_ok={ok}")
        except Exception as ex:
            rec["pdf_ok"] = False
            rec["pdf_error"] = str(ex)
            print(f"  -> {arxiv_id}  PDF download failed: {ex}")
        meta_lines.append(rec)
        time.sleep(1)

    (out / "metadata.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in meta_lines), encoding="utf-8")
    (out / "resolve_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== DONE ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
