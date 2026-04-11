from __future__ import annotations
 
import csv
import gc
import io
import logging
import os
import re
import time
from typing import Any
from urllib.parse import urljoin
 
import nltk
import requests
from bs4 import BeautifulSoup
from goose3 import Goose
from PIL import Image
 
nltk.download("wordnet", quiet=True)
from nltk.corpus import wordnet 



#from soft404 import Soft404Classifier
from soft404_duplicate import is_dead
from soft404_duplicate import added_url
import concurrent.futures
import nltk
from nltk.corpus import wordnet
################


logger = logging.getLogger(__name__)
_GOOSE = Goose()
REQUEST_TIMEOUT = 30

## Text sanitization
def is_meaningful_image_name(name: str) -> bool:
    """Return True if any underscore-separated token in *name* is a WordNet entry."""
    stem = os.path.splitext(name)[0]          # drop extension before splitting
    return any(wordnet.synsets(tok) for tok in stem.split("_") if tok)

def sanitize_csv_text(text: str, max_words: int = 100) -> str:
    """Strip CSV-hostile characters and truncate to *max_words* words."""
    text = re.sub(r"[,\t;|: ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(text.split()[:max_words])
 
## feature extraction

def get_file_size(url: str) -> int:
    """HEAD *url* and return Content-Length (0 on any failure).
    has to be removed in future versions."""
    try:
        resp = requests.head(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return int(resp.headers.get("content-length", 0))
    except requests.exceptions.RequestException:
        pass
    return 0

SOFT404_KEYWORDS: list[str] = [
    "404", "error", "found", "page", "file", "resource", "url",
    "access denied", "forbidden", "unavailable", "gone", "redirect",
    "temporary", "permanent", "server", "client", "browser", "network",
    "dns", "typo", "misspelling", "stale", "broken", "dead", "obsolete",
    "expired", "invalid", "malformed", "corrupt", "unreachable", "deleted",
]
 
_KEYWORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kw) for kw in SOFT404_KEYWORDS) + r")\b",
    re.IGNORECASE,
)
 


def extract_text_features(
    soup: BeautifulSoup,
    raw_html: bytes | str,
    *,
    max_images: int = 5,
    max_img_name_len: int = 25,
) -> dict[str, Any]:
    """
    Extract NLP / content-level features.
 
    Returns
    -------
    dict with keys:
        title, article_text, paragraph_text, image_names, meaningful_image_names
    """
    # --- title ---
    title_tag = soup.find("title")
    title = sanitize_csv_text(title_tag.text if title_tag else "")
 
    # --- goose article ---
    html_bytes = raw_html if isinstance(raw_html, bytes) else raw_html.encode()
    article = _GOOSE.extract(raw_html=html_bytes)
    article_text = sanitize_csv_text(article.cleaned_text[:200])
 
    # --- paragraph text ---
    paras = [p.get_text() for p in soup.find_all("p")]
    paragraph_text = sanitize_csv_text(" ".join(paras))
 
    # --- image names (top-N) ---
    def _name(img_tag) -> str:
        raw = os.path.basename(img_tag.get("src", ""))
        return raw[:max_img_name_len] if len(raw) > max_img_name_len else raw
 
    images = soup.find_all("img")[:max_images]
    image_names = [_name(img) for img in images if img.get("src")]
    meaningful_image_names = [n for n in image_names if is_meaningful_image_name(n)]
 
    return {
        "title": title,
        "article_text": article_text,
        "paragraph_text": paragraph_text,
        "image_names": image_names,
        "meaningful_image_names": meaningful_image_names,
    }
 

def extract_structural_features(
    url: str,
    response: requests.Response,
    soup: BeautifulSoup,
    load_time: float,
) -> dict[str, Any]:
    """
    Extract byte-size / structural features.
 
    Returns
    -------
    dict with keys:
        size_total, load_time, image_count, image_unreachable,
        image_size_total, ratio, average_word_length,
        script_size, css_size, matched_keywords, number_of_words
    """
    size_total = len(response.content)
    base_url = response.url
 
    # --- image sizes ---
    image_size_total = 0
    image_unreachable = 0
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src")
        if not src:
            continue
        abs_url = urljoin(base_url, src)
        try:
            img_resp = requests.get(abs_url, stream=True, timeout=REQUEST_TIMEOUT)
            img_data = io.BytesIO(img_resp.content)
            Image.open(img_data)               # validate it is an image
            image_size_total += img_data.getbuffer().nbytes
        except Exception:
            image_unreachable += 1
 
    # --- script sizes ---
    script_size = sum(
        get_file_size(url + t["src"] if not t["src"].startswith("http") else t["src"])
        for t in soup.find_all("script")
        if t.get("src")
    )
 
    # --- css sizes ---
    css_size = sum(
        get_file_size(url + l["href"] if not l["href"].startswith("http") else l["href"])
        for l in soup.find_all("link", rel="stylesheet")
        if l.get("href")
    )
 
    # --- keyword matches ---
    matched_keywords = len(soup.find_all(string=_KEYWORD_PATTERN))
 
    # --- word stats ---
    all_words = soup.find_all(string=True)
    word_length_sum = sum(len(w) for w in all_words)
    average_word_length = word_length_sum / len(all_words) if all_words else 0
 
    title_tag = soup.find("title")
    number_of_words = len(title_tag.text.split()) if title_tag else 0
 
    content_bytes = size_total - image_size_total
    ratio = content_bytes / size_total if size_total else 0.0
 
    return {
        "size_total": size_total,
        "load_time": load_time,
        "image_count": len(soup.find_all("img")),
        "image_unreachable": image_unreachable,
        "image_size_total": image_size_total,
        "ratio": ratio,
        "average_word_length": average_word_length,
        "script_size": script_size,
        "css_size": css_size,
        "matched_keywords": matched_keywords,
        "number_of_words": number_of_words,
    }
 


##### CSV writers 

TEXT_COLUMNS = [
    "url", "title", "article_text", "paragraph_text",
    "meaningful_image_names", "response_status_code", "is_dead", "prob",
]
 
STRUCTURAL_COLUMNS = [
    "url", "size_total", "load_time", "image_count", "image_unreachable",
    "average_word_length", "ratio", "image_size_total", "matched_keywords",
    "number_of_words", "script_size", "css_size",
    "response_status_code", "is_dead", "prob",
]
 
FULL_COLUMNS = STRUCTURAL_COLUMNS + [
    "title", "article_text", "paragraph_text", "meaningful_image_names",
]
 
ERROR_COLUMNS = ["url", "error_text"]
 
 
 
def write_header(filepath: str, columns: list[str]) -> None:
    """Write (or overwrite) a CSV with only the header row."""
    with open(filepath, "w", newline="") as fh:
        csv.writer(fh).writerow(columns)
 
 
def append_row(filepath: str, row: dict[str, Any], columns: list[str]) -> None:
    """Append one row to *filepath*, selecting fields by *columns* order."""
    with open(filepath, "a", newline="") as fh:
        csv.writer(fh).writerow([row.get(col, "") for col in columns])