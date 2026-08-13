import csv
import time
import requests
from pathlib import Path

API_KEY = "Q2bbP41q4woiJUZBbtVSpU"
BASE_URL = "https://api.openalex.org"
PER_PAGE = 200  # max allowed by OpenAlex
SLEEP_BETWEEN_REQUESTS = 0.1  # stay comfortably under 100 req/sec

def resolve_source_id(journal_name: str, verbose=False) -> dict | None:
    """Look up a journal name and return its OpenAlex source record."""
    resp = requests.get(
        f"{BASE_URL}/sources",
        params={"search": journal_name, "per_page": 5, "api_key": API_KEY},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        print(f"  [!] No match found for '{journal_name}'")
        return None

    top = results[0]
    if verbose:
        print(f"  '{journal_name}' -> {top['display_name']} ({top['id']})")
    if len(results) > 1:
        for alt in results[1:3]:
            if verbose:
                print(f"      (alt: {alt['display_name']}, {alt['id']})")
    return top


def get_year_counts(source_id: str) -> dict:
    """One call per journal: returns {year: count} via group_by."""
    resp = requests.get(
        f"{BASE_URL}/works",
        params={
            "filter": f"primary_location.source.id:{source_id}",
            "group_by": "publication_year",
            "api_key": API_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    groups = resp.json().get("group_by", [])
    return {int(g["key"]): g["count"] for g in groups if g["key"].isdigit()}


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Turn OpenAlex's word->positions inverted index back into plain text."""
    if not inverted_index:
        return ""
    position_map = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    return " ".join(position_map[i] for i in sorted(position_map))


def fetch_works_for_source_year(source_id: str, year: int):
    """Yield all works for one source (journal) and one publication year."""
    cursor = "*"
    while cursor:
        params = {
            "filter": f"primary_location.source.id:{source_id},publication_year:{year}",
            "per_page": PER_PAGE,
            "cursor": cursor,
            "api_key": API_KEY,
        }
        resp = requests.get(f"{BASE_URL}/works", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for work in data.get("results", []):
            yield work

        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(SLEEP_BETWEEN_REQUESTS)


def extract_row(work: dict, journal_name: str) -> dict:
    """Flatten one OpenAlex work record into a CSV row."""
    authors = "; ".join(
        auth["author"]["display_name"]
        for auth in work.get("authorships", [])
        if auth.get("author")
    )
    return {
        "journal": journal_name,
        "openalex_id": work.get("id", ""),
        "doi": work.get("doi", ""),
        "title": work.get("title") or "",
        "year": work.get("publication_year", ""),
        "publication_date": work.get("publication_date", ""),
        "authors": authors,
        "cited_by_count": work.get("cited_by_count", ""),
        "is_oa": work.get("open_access", {}).get("is_oa", ""),
        "abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
    }
