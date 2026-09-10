"""Verify every reference against Crossref and write references.bib.

    python paper/verify_refs.py --refs paper/refs.json --overrides paper/doi_overrides.json \
        --bib paper/references.bib --report paper/refs_report.csv

Three routes to a verified entry:
  1. Crossref title search (rows=3): best hit with title similarity >= 0.85 and year within 1 -> DOI.
  2. A DOI from doi_overrides.json: fetched from Crossref works/{doi}; accepted only if the
     returned title matches ours (similarity >= 0.85). Used when the search matches the wrong record
     (a review, a preprint) or misses a known paper.
  3. No DOI by design (arXiv-only, NeurIPS/ICLR/PMLR proceedings): the entry carries an arXiv id or a
     URL and is marked OK-NODOI with the reason from the overrides file.
Anything else is CHECK and gets no DOI. Nothing is invented.
"""
import argparse
import csv
import difflib
import json
import re
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "dmthd-refs/0.2 (mailto:dmthd-thesis@example.org)"}
SEARCH = "https://api.crossref.org/works?rows=3&query.bibliographic={q}&mailto=dmthd-thesis@example.org"
WORK = "https://api.crossref.org/works/{doi}"


def norm(t):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split())


def similarity(a, b):
    """Title similarity in [0, 1]. Crossref sometimes stores only the main title (e.g. "Ex Machina"
    for "Ex Machina: Personal Attacks Seen at Scale"); a record whose title is a prefix of ours,
    at least 8 characters long, counts as a full match."""
    na, nb = norm(a), norm(b)
    if len(nb) >= 8 and (na.startswith(nb) or nb.startswith(na)):
        return 1.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40) as r:
        return json.load(r)


def year_of(it):
    for k in ("published-print", "published-online", "issued", "created"):
        parts = (it.get(k) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


def describe(it, title):
    t = (it.get("title") or [""])[0]
    return {"doi": it.get("DOI"), "title": t, "container": (it.get("container-title") or [""])[0],
            "year": year_of(it), "score": similarity(title, t)}


def search(title):
    items = get_json(SEARCH.format(q=urllib.parse.quote(title)))["message"]["items"]
    cands = [describe(it, title) for it in items]
    return max(cands, key=lambda c: c["score"]) if cands else None


def lookup(doi, title):
    return describe(get_json(WORK.format(doi=urllib.parse.quote(doi)))["message"], title)


def bibtex(e, doi, url=None):
    fields = {"title": "{" + e["title"] + "}", "author": e["author"], "year": str(e["year"])}
    fields["journal" if e["type"] == "article" else "booktitle" if e["type"] == "inproceedings" else "howpublished"] = e["venue"]
    for k in ("volume", "number", "pages"):
        if e.get(k):
            fields[k] = e[k]
    if doi:
        fields["doi"] = doi
    elif e.get("arxiv"):
        fields["eprint"], fields["archivePrefix"] = e["arxiv"], "arXiv"
    if url:
        fields["url"] = url
    body = ",\n".join(f"  {k} = {v}" if v.startswith("{") else f"  {k} = {{{v}}}" for k, v in fields.items())
    return f"@{e['type']}{{{e['key']},\n{body}\n}}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refs", default="paper/refs.json")
    ap.add_argument("--overrides", default="paper/doi_overrides.json")
    ap.add_argument("--bib", default="paper/references.bib")
    ap.add_argument("--report", default="paper/refs_report.csv")
    ap.add_argument("--min_score", type=float, default=0.85)
    args = ap.parse_args()
    refs = json.load(open(args.refs, encoding="utf-8"))
    ov = json.load(open(args.overrides, encoding="utf-8"))
    overrides = {k: v for k, v in ov.items() if not k.startswith("_")}
    no_doi = ov.get("_no_doi_by_design", {})

    rows, bib = [], []
    for e in refs:
        key, title = e["key"], e["title"]
        status, doi, url, best = "CHECK", None, None, None
        try:
            if key in overrides:
                best = lookup(overrides[key], title)
                if best["score"] >= args.min_score:
                    status, doi = "OK-DOI-OVERRIDE", best["doi"]
            else:
                best = search(title)
                if best and best["score"] >= args.min_score and (best["year"] is None or abs(best["year"] - int(e["year"])) <= 1):
                    status, doi = "OK", best["doi"]
        except Exception as ex:
            best = {"doi": None, "title": f"error: {str(ex)[:60]}", "container": "", "year": None, "score": 0.0}
        if status == "CHECK" and key in no_doi:
            status = "OK-NODOI"
            url = no_doi[key] if no_doi[key].startswith("http") else None
        rows.append({"key": key, "status": status, "score": round(best["score"], 3) if best else 0, "doi": doi or "",
                     "note": no_doi.get(key, ""), "crossref_title": (best or {}).get("title", "")[:90],
                     "crossref_container": (best or {}).get("container", "")[:60], "crossref_year": (best or {}).get("year", ""),
                     "our_year": e["year"], "our_venue": e["venue"][:60]})
        bib.append(bibtex(e, doi, url))
        print(f"{status:16s} {round(best['score'], 2) if best else 0:>4}  {key:26s} {doi or url or '(arXiv/none)'}", flush=True)
        time.sleep(0.3)

    with open(args.bib, "w", encoding="utf-8") as f:
        f.write("% Generated by paper/verify_refs.py from paper/refs.json; every DOI verified against Crossref.\n\n" + "\n".join(bib))
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\nsummary:", counts, "->", args.report)


if __name__ == "__main__":
    main()
