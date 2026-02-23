from __future__ import annotations

from datetime import datetime
from typing import List

import requests

from .schemas import PaperRecord


class AbstractCollector:
    def fetch_from_arxiv(self, query: str, max_results: int = 50) -> List[PaperRecord]:
        url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        xml = response.text

        records: List[PaperRecord] = []
        entries = xml.split("<entry>")[1:]
        for entry in entries:
            title = _between(entry, "<title>", "</title>").replace("\n", " ").strip()
            summary = _between(entry, "<summary>", "</summary>").replace("\n", " ").strip()
            paper_id = _between(entry, "<id>", "</id>").strip()
            published = _between(entry, "<published>", "</published>").strip()
            year = None
            if published:
                year = datetime.fromisoformat(published.replace("Z", "+00:00")).year
            if title and summary:
                records.append(
                    PaperRecord(
                        paper_id=paper_id,
                        title=title,
                        abstract=summary,
                        source="arXiv",
                        year=year,
                    )
                )
        return records

    def fetch_from_crossref(self, query: str, rows: int = 50) -> List[PaperRecord]:
        url = "https://api.crossref.org/works"
        params = {"query": query, "rows": rows, "select": "DOI,title,abstract,created"}
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])

        records: List[PaperRecord] = []
        for item in items:
            abstract = item.get("abstract")
            if not abstract:
                continue
            title_list = item.get("title", [])
            if not title_list:
                continue
            year = None
            created = item.get("created", {}).get("date-parts", [])
            if created and created[0]:
                year = created[0][0]
            records.append(
                PaperRecord(
                    paper_id=item.get("DOI", ""),
                    doi=item.get("DOI"),
                    title=title_list[0],
                    abstract=_strip_jats_tags(abstract),
                    source="Crossref",
                    year=year,
                )
            )
        return records


def _between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0]


def _strip_jats_tags(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", text).strip()
