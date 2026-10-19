# -*- coding: utf-8 -*-
"""arXiv 关键词批量拉取变化检测论文，扩充语料到 ~130-150 篇。

用法：python3 expand_papers.py --out /path/to/corpus --target 140
"""
from __future__ import annotations

import argparse
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

KEYWORDS = [
    "remote sensing change detection",
    "building change detection deep learning",
    "semantic change detection remote sensing",
    "bitemporal change detection",
    "change detection transformer",
    "change detection siamese network",
]


def fetch(url: str, timeout: int = 40) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        eid = e.findtext("a:id", default="", namespaces=NS)
        title = re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=NS)).strip()
        summary = re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=NS)).strip()
        pub = e.findtext("a:published", default="", namespaces=NS)[:4]
        arxiv_id = eid.split("/abs/")[-1] if "/abs/" in eid else ""
        out.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "abstract": summary,
            "year": pub,
            "pdf": eid.replace("/abs/", "/pdf/") + ".pdf" if arxiv_id else "",
        })
    return out


def search(keyword: str, start: int = 0, max_results: int = 50) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode({
        "search_query": f'all:"{keyword}"',
        "start": start,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    return parse_entries(fetch(url))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=140)
    args = ap.parse_args()

    out = Path(args.out)
    pdf_dir = out / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    seen: dict[str, dict] = {}
    for kw in KEYWORDS:
        print(f"=== search: {kw} ===", flush=True)
        for start in (0, 50):
            if len(seen) >= args.target:
                break
            try:
                entries = search(kw, start=start, max_results=50)
            except Exception as ex:
                print(f"  search failed {kw}@{start}: {ex}", flush=True)
                time.sleep(5)
                continue
            for e in entries:
                if e["arxiv_id"] and e["arxiv_id"] not in seen:
                    seen[e["arxiv_id"]] = e
            print(f"  collected {len(seen)} unique so far", flush=True)
            time.sleep(3)
        if len(seen) >= args.target:
            break

    papers = list(seen.values())
    print(f"=== total unique: {len(papers)} ===", flush=True)

    meta_lines = []
    downloaded = 0
    for i, p in enumerate(papers, 1):
        aid = p["arxiv_id"].replace("/", "_")
        pdf_path = pdf_dir / f"{aid}.pdf"
        rec = {
            "arxiv_id": p["arxiv_id"],
            "title": p["title"],
            "year": p["year"],
            "abstract": p["abstract"],
            "category": "expanded",
            "source": "arxiv_keyword_search",
        }
        try:
            if not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
                data = urllib.request.urlopen(
                    urllib.request.Request(p["pdf"], headers=UA), timeout=90).read()
                pdf_path.write_bytes(data)
            ok = pdf_path.stat().st_size > 10_000 and pdf_path.read_bytes()[:4] == b"%PDF"
            rec["pdf_bytes"] = pdf_path.stat().st_size
            rec["pdf_ok"] = ok
            downloaded += 1 if ok else 0
            if i % 20 == 0 or i == len(papers):
                print(f"  [{i}/{len(papers)}] {aid} pdf_ok={ok}", flush=True)
        except Exception as ex:
            rec["pdf_ok"] = False
            rec["pdf_error"] = str(ex)
        meta_lines.append(rec)
        time.sleep(0.5)

    (out / "metadata_expanded.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in meta_lines), encoding="utf-8")
    (out / "expand_report.json").write_text(json.dumps({
        "unique_papers": len(papers), "downloaded": downloaded,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== DONE: {len(papers)} unique, {downloaded} downloaded ===", flush=True)


if __name__ == "__main__":
    main()
