# -*- coding: utf-8 -*-
"""健壮的合并论文下载器：关键词搜索 + 种子标题，429 退避重试，增量写盘。

用法：python3 fetch_all.py --out /path/to/corpus --manifest papers_manifest.json --target 120
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

UA = {"User-Agent": "rscd-corpus/1.0 (research assistant; contact: local)"}
ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

KEYWORDS = [
    "remote sensing change detection",
    "building change detection deep learning",
    "semantic change detection remote sensing",
    "bitemporal change detection",
    "change detection transformer",
    "change detection siamese network",
    "change detection mamba",
    "change detection diffusion model",
    "change detection foundation model",
    "remote sensing change detection dataset",
    "cropland change detection",
    "change detection weakly supervised",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _get(url: str, timeout: int = 60, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 500) and attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"  HTTP {e.code}, retry in {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(8)
                continue
            raise
    raise RuntimeError("unreachable")


def parse_entries(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        eid = e.findtext("a:id", default="", namespaces=NS)
        title = re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=NS)).strip()
        summary = re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=NS)).strip()
        pub = e.findtext("a:published", default="", namespaces=NS)[:4]
        arxiv_id = eid.split("/abs/")[-1] if "/abs/" in eid else ""
        out.append({"arxiv_id": arxiv_id, "title": title, "abstract": summary, "year": pub,
                    "pdf": f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""})
    return out


def search(keyword: str, start: int, max_results: int = 50) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode({
        "search_query": f'all:"{keyword}"', "start": start,
        "max_results": max_results, "sortBy": "relevance"})
    return parse_entries(_get(url).decode("utf-8"))


def search_title(title: str) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode({
        "search_query": f'ti:"{title[:200]}"', "max_results": 3})
    return parse_entries(_get(url).decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--target", type=int, default=120)
    args = ap.parse_args()

    out = Path(args.out)
    pdf_dir = out / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.jsonl"
    done_ids = set()
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line).get("arxiv_id", ""))

    seen: dict[str, dict] = {}

    # 1) 关键词搜索
    for kw in KEYWORDS:
        if len(seen) >= args.target:
            break
        for start in (0, 50):
            if len(seen) >= args.target:
                break
            try:
                entries = search(kw, start)
            except Exception as ex:
                print(f"search failed {kw}@{start}: {ex}", flush=True)
                time.sleep(15)
                continue
            for e in entries:
                if e["arxiv_id"] and e["arxiv_id"] not in seen:
                    seen[e["arxiv_id"]] = e
            print(f"  [{kw}] unique={len(seen)}", flush=True)
            time.sleep(4)

    # 2) 种子标题（补漏，仅当不足）
    if args.manifest and Path(args.manifest).exists() and len(seen) < args.target:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for p in manifest["papers"]:
            if len(seen) >= args.target:
                break
            try:
                entries = search_title(p["title"])
                if entries:
                    # 最相似
                    nt = norm(p["title"])
                    best = max(entries, key=lambda e: (
                        0.5 if (nt and nt in norm(e["title"])) else 0) +
                        (1.0 if (nt and norm(e["title"]) == nt) else 0))
                    if best["arxiv_id"] and best["arxiv_id"] not in seen:
                        best["aka"] = p.get("aka", "")
                        seen[best["arxiv_id"]] = best
            except Exception as ex:
                print(f"seed search failed {p['title'][:40]}: {ex}", flush=True)
            time.sleep(4)

    print(f"=== collected {len(seen)} unique, will download ===", flush=True)

    # 3) 下载 PDF + 增量写 metadata
    n_new = 0
    for i, (aid, p) in enumerate(seen.items(), 1):
        if aid in done_ids:
            continue
        pdf_path = pdf_dir / f"{aid.replace('/', '_')}.pdf"
        rec = {"arxiv_id": aid, "title": p["title"], "year": p.get("year"),
               "abstract": p.get("abstract", ""), "aka": p.get("aka", ""),
               "category": p.get("category", "keyword")}
        try:
            if not (pdf_path.exists() and pdf_path.stat().st_size > 10_000):
                data = _get(p["pdf"], timeout=120)
                if data[:4] != b"%PDF":
                    raise RuntimeError("not a pdf")
                pdf_path.write_bytes(data)
            rec["pdf_bytes"] = pdf_path.stat().st_size
            rec["pdf_ok"] = pdf_path.read_bytes()[:4] == b"%PDF"
            n_new += 1 if rec["pdf_ok"] else 0
        except Exception as ex:
            rec["pdf_ok"] = False
            rec["pdf_error"] = str(ex)[:200]
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 10 == 0 or i == len(seen):
            print(f"  [{i}/{len(seen)}] {aid} ok={rec.get('pdf_ok')}", flush=True)
        time.sleep(3)

    print(f"=== DONE: {len(seen)} total, {n_new} new downloaded ===", flush=True)


if __name__ == "__main__":
    main()
