"""Download engineering drawings from open data sources.

This script queries NASA's image library and Wikimedia Commons for
engineering drawing imagery. Downloaded files are converted to PNG, with
optional skeletonization and graph extraction for later machine-learning
use.  The script stores images beneath a ``data/`` directory, grouped by
query term.

The search results are restricted to content returned by the respective
public APIs.  Users must verify the licensing requirements of each image
before using it in derivative works.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict

import requests
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
from skimage.morphology import skeletonize
import networkx as nx

SESSION = requests.Session()
# Respect system proxy settings by default; overridden with ``--no-proxy``.
SESSION.trust_env = True


def http_get(url: str, **kwargs) -> requests.Response:
    """HTTP GET using the configured ``SESSION``."""
    try:
        resp = SESSION.get(url, **kwargs)
    except requests.exceptions.ProxyError as e:  # pragma: no cover - network failure
        raise RuntimeError(
            "Proxy connection failed; use --no-proxy or check network access"
        ) from e
    resp.raise_for_status()
    return resp

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PNG_DIR = DATA_DIR / "png"
SKEL_DIR = DATA_DIR / "skeleton"
GRAPH_DIR = DATA_DIR / "graph"
METADATA_DIR = DATA_DIR / "metadata"

for d in [RAW_DIR, PNG_DIR, SKEL_DIR, GRAPH_DIR, METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------
# Search helpers
# ----------------------------------------------------------------------------

NASA_API = "https://images-api.nasa.gov/search"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"


def search_nasa(query: str, limit: int) -> List[Dict]:
    """Return a list of image metadata dicts from the NASA image API."""
    params = {"q": query, "media_type": "image", "page": 1}
    items: List[Dict] = []
    while len(items) < limit:
        resp = http_get(NASA_API, params=params, timeout=30)
        data = resp.json()
        collection = data.get("collection", {})
        for item in collection.get("items", []):
            links = item.get("links", [])
            if not links:
                continue
            href = links[0].get("href")
            if not href:
                continue
            items.append({
                "url": href,
                "source": "nasa",
                "title": item.get("data", [{}])[0].get("title", ""),
                "description": item.get("data", [{}])[0].get("description", ""),
                "license": "public domain"
            })
            if len(items) >= limit:
                break
        if len(items) < limit and collection.get("links"):
            params["page"] += 1
        else:
            break
    return items


def search_wikimedia(query: str, limit: int) -> List[Dict]:
    """Return a list of image metadata dicts from Wikimedia Commons."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "imageinfo",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": limit,
        "iiprop": "url|mime",
    }
    resp = http_get(COMMONS_API, params=params, timeout=30)
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    results: List[Dict] = []
    for page in pages.values():
        info = page.get("imageinfo", [{}])[0]
        url = info.get("url")
        if not url:
            continue
        results.append({
            "url": url,
            "source": "wikimedia",
            "title": page.get("title", ""),
            "mime": info.get("mime", ""),
            "license": info.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
        })
        if len(results) >= limit:
            break
    return results


# ----------------------------------------------------------------------------
# Conversion utilities
# ----------------------------------------------------------------------------


def download_file(url: str, dest: Path) -> Path:
    resp = http_get(url, timeout=60)
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


def convert_to_png(src: Path, dest: Path) -> Path:
    """Convert ``src`` to PNG at ``dest``. Supports images & PDFs."""
    ext = src.suffix.lower()
    if ext == ".pdf":
        images = convert_from_path(src)
        if not images:
            raise RuntimeError(f"No pages in {src}")
        images[0].save(dest)
    elif ext in {".jpg", ".jpeg", ".png", ".bmp"}:
        Image.open(src).save(dest)
    elif ext in {".tif", ".tiff"}:
        Image.open(src).convert("RGB").save(dest)
    else:
        raise ValueError(f"Unsupported file type: {src.suffix}")
    return dest


def skeletonize_image(png_path: Path, dest: Path) -> Path:
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    skel = skeletonize(bw // 255).astype(np.uint8) * 255
    cv2.imwrite(str(dest), skel)
    return dest


def skeleton_to_graph(skel_path: Path, dest: Path) -> Path:
    """Convert a skeleton image to a graph serialized as GraphML."""
    img = cv2.imread(str(skel_path), cv2.IMREAD_GRAYSCALE)
    points = np.argwhere(img > 0)
    G = nx.Graph()
    for y, x in points:
        G.add_node((y, x))
    for y, x in points:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (ny, nx_) in G:
                    G.add_edge((y, x), (ny, nx_))
    nx.write_graphml(G, dest)
    return dest


# ----------------------------------------------------------------------------
# Main CLI
# ----------------------------------------------------------------------------


def process_query(query: str, limit: int) -> None:
    query_sanitized = "_".join(query.split())
    raw_dir = RAW_DIR / query_sanitized
    png_dir = PNG_DIR / query_sanitized
    skel_dir = SKEL_DIR / query_sanitized
    graph_dir = GRAPH_DIR / query_sanitized
    for d in [raw_dir, png_dir, skel_dir, graph_dir]:
        d.mkdir(parents=True, exist_ok=True)

    metadata: List[Dict] = []
    results = search_nasa(query, limit) + search_wikimedia(query, limit)

    for idx, item in enumerate(results):
        url = item["url"]
        ext = Path(url).suffix or ".jpg"
        raw_path = raw_dir / f"{idx:05d}{ext}"
        download_file(url, raw_path)
        png_path = png_dir / f"{idx:05d}.png"
        try:
            convert_to_png(raw_path, png_path)
        except Exception as e:
            print(f"Failed to convert {raw_path}: {e}")
            continue
        skel_path = skel_dir / f"{idx:05d}.png"
        skeletonize_image(png_path, skel_path)
        graph_path = graph_dir / f"{idx:05d}.graphml"
        skeleton_to_graph(skel_path, graph_path)
        item["raw_path"] = str(raw_path)
        item["png_path"] = str(png_path)
        metadata.append(item)

    with open(METADATA_DIR / f"{query_sanitized}.json", "w") as f:
        json.dump(metadata, f, indent=2)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download engineering drawings")
    parser.add_argument("queries", nargs="+", help="search terms")
    parser.add_argument("--limit", type=int, default=50, help="images per source")
    parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="ignore system proxy settings for HTTP requests",
    )
    args = parser.parse_args(argv)

    SESSION.trust_env = not args.no_proxy

    for query in args.queries:
        process_query(query, args.limit)


if __name__ == "__main__":
    main()
