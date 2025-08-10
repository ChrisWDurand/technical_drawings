# Technical Drawing Collector

This project provides a Docker-based environment for gathering internet images of
mechanical engineering drawings (e.g., work stands, jigs and other production
tools). Downloaded images are converted into PNG files ready for use in
computer-vision training pipelines. A [CVAT](https://github.com/opencv/cvat)
service is included for annotating the collected data.

## Structure
- `download_drawings.py` – script to download images and convert them to PNG.
- `Dockerfile` – builds the image-collection container.
- `docker-compose.yml` – runs the collector and a CVAT service sharing the same
  `data/` volume.
- `requirements.txt` – Python dependencies for the collector script.

## Usage
1. Build the collector container:
   ```bash
   docker compose build
   ```
2. Download drawings (stored under `data/`):
   ```bash
   docker compose run --rm collector "mechanical engineering drawing" "work stand jig" --limit 5
   ```
3. Launch CVAT for annotation:
   ```bash
   docker compose up cvat
   ```
   Open [http://localhost:8080](http://localhost:8080) in a browser and create a
   task using images from the shared `data/png` directory.

The resulting PNG images and annotations can be used to train a computer vision
model.
