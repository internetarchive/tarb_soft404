'''
Unified dataset builder for the soft-404 pipeline.
 
Usage
-----
python make_dataset.py \\
    --input      urls.csv        \\   # input CSV
    --output     out.csv         \\   # main output CSV
    --url-col    0               \\   # column index (or name) that holds URLs
    --mode       full            \\   # text | structural | full  (default: full)
    --workers    32              \\   # ThreadPoolExecutor concurrency
    [--error-output errors.csv]  \\   # if given, failed rows go here
    [--normalize-urls]               # run added_url() on every URL before fetching
                                     # (replicates Context_extractor_error behaviour)
 
Modes
-----
text         Extracts NLP/content features only (title, article, paragraphs,
             image names, is_dead).  Fast — skips per-image HTTP requests.
 
structural   Extracts byte-size/structural features only (size, load time,
             images, scripts, CSS, keyword counts).  Slow due to image fetching.
 
full         Both text + structural features combined

'''

from __future__ import annotations
 
import argparse
import gc
import logging
import sys
import time
import concurrent.futures
from typing import Callable

 
import pandas as pd
import requests
from bs4 import BeautifulSoup
 
from soft404_duplicate import is_dead, added_url          
from utils import (
    ERROR_COLUMNS,
    FULL_COLUMNS,
    STRUCTURAL_COLUMNS,
    TEXT_COLUMNS,
    append_row,
    extract_structural_features,
    extract_text_features,
    write_header,
    REQUEST_TIMEOUT,
)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

#`------------------
from urllib.parse import urljoin
import pandas as pd
from requests.exceptions import ConnectionError
from urllib3.exceptions import NewConnectionError

from soft404_duplicate import added_url , is_dead
import concurrent.futures
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')


from src.utils import is_meaningful ,sanitize_csv_text,get_website_details



## Per URL processing function

 
def process_url(
    url: str,
    *,
    mode: str,
    url_error: bool,
    output_file: str,
    error_file: str | None,
    columns: list[str],
) -> None:
    """Fetch *url*, extract features according to *mode*, write to CSV."""
 
    if url_error:
        url = added_url(url)
 
    try:
        t0 = time.time()
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        load_time = time.time() - t0
 
        raw = response.content
        soup = BeautifulSoup(raw, "html.parser")
 
        dead = is_dead(url)
        status = response.status_code
 
        row: dict = {"url": url, "response_status_code": status, "is_dead": dead, "prob": 0}
 
        if mode in ("text", "full"):
            row.update(extract_text_features(soup, raw))
 
        if mode in ("structural", "full"):
            row.update(extract_structural_features(url, response, soup, load_time))
 
        # Flatten list fields to a space-separated string for CSV compatibility
        for key in ("image_names", "meaningful_image_names"):
            if isinstance(row.get(key), list):
                row[key] = " ".join(row[key])
 
        append_row(output_file, row, columns)
 
    except Exception as exc:
        logger.warning("Error processing %s: %s", url, exc)
        if error_file:
            append_row(
                error_file,
                {"url": url, "error_text": str(exc)},
                ERROR_COLUMNS,
            )
 

 

 
 
def _resolve_columns(mode: str) -> list[str]:
    return {"text": TEXT_COLUMNS, "structural": STRUCTURAL_COLUMNS, "full": FULL_COLUMNS}[mode]
 
 
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a soft-404 feature dataset from a CSV of URLs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",          required=True,  help="Input CSV file path")
    parser.add_argument("--output",         required=True,  help="Output CSV file path")
    parser.add_argument("--url-col",        default="0",    help="Column index or name for URLs (default: 0)")
    parser.add_argument("--mode",           default="full",
                        choices=["text", "structural", "full"],
                        help="Feature extraction mode (default: full)")
    parser.add_argument("--workers",        type=int, default=32, help="Thread-pool size (default: 32)")
    parser.add_argument("--error-output",   default=None,   help="CSV for failed URLs (optional)")
    parser.add_argument("--normalize-urls", action="store_true",
                        help="Pre-process URLs with added_url() before fetching")
    args = parser.parse_args(argv)
 
    # --- load URLs ---
    df = pd.read_csv(args.input)
    try:
        col = int(args.url_col)           # integer index
        urls = df.iloc[:, col].tolist()
    except ValueError:
        urls = df[args.url_col].tolist()  # column name
 
    urls = [str(u).strip() for u in urls if pd.notna(u)]
    logger.info("Loaded %d URLs from %s", len(urls), args.input)
 
    # --- prepare output files ---
    columns = _resolve_columns(args.mode)
    write_header(args.output, columns)
    if args.error_output:
        write_header(args.error_output, ERROR_COLUMNS)
 
    # --- process concurrently ---
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_url,
                url,
                mode=args.mode,
                normalize=args.normalize_urls,
                output_file=args.output,
                error_file=args.error_output,
                columns=columns,
            ): url
            for url in urls
        }
 
        for fut in concurrent.futures.as_completed(futures):
            processed += 1
            if processed % 50 == 0:
                gc.collect()
            if processed % 100 == 0 or processed == len(urls):
                logger.info("Progress: %d / %d", processed, len(urls))
 
            # surface unexpected exceptions from the future itself
            try:
                fut.result()
            except Exception as exc:
                logger.error("Unhandled future error for %s: %s", futures[fut], exc)
 
    logger.info("Done. Output written to %s", args.output)
    if args.error_output:
        logger.info("Errors written to %s", args.error_output)
 
 
if __name__ == "__main__":
    main()
 