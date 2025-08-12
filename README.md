# Technical Drawing Collector

This project provides a Docker-based environment for gathering internet images of
mechanical engineering drawings (e.g., work stands, jigs and other production
tools). The `download_drawings.py` script queries open sources such as NASA's
image library and Wikimedia Commons. Files are converted to PNG, skeletonized
and translated into simple graphs for experimentation with computer-vision and
graph-learning models. A [CVAT](https://github.com/opencv/cvat) service is
included for annotating the collected data. The downloader uses system proxy
settings by default; pass ``--no-proxy`` to bypass proxies if necessary.

## Structure
- `download_drawings.py` – search NASA and Wikimedia for images, convert to PNG,
  produce skeletons and graph representations.
- `Dockerfile` – builds the image-collection container.
- `docker-compose.yml` – runs the collector and a CVAT service sharing the same
  `data/` volume.
- `requirements.txt` – Python dependencies for the collector script.

The script stores outputs under `data/` with subfolders:
- `raw/` – original downloads grouped by search term.
- `png/` – converted PNG images.
- `skeleton/` – skeletonized PNGs.
- `graph/` – GraphML files derived from skeletons.
- `metadata/` – JSON files describing the retrieved images and their licenses.

## Usage
1. Build the collector container:
   ```bash
   docker compose build
   ```
2. Download drawings (stored under `data/`):
   ```bash
   docker compose run --rm collector "mechanical engineering drawing" "work stand jig" --limit 5
   ```
   Append `--no-proxy` if your environment sets proxies and they should be
   ignored for the download.
3. Launch CVAT for annotation:
   ```bash
   docker compose up cvat
   ```
   Open [http://localhost:8080](http://localhost:8080) in a browser and create a
   task using images from the shared `data/png` directory.

The resulting PNG images, skeletons and graphs can be used to train a computer
vision model. Always verify the license terms of downloaded content before
redistributing or using the data in derived works.
