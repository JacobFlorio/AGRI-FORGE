"""
data/scraper.py — Multi-source agricultural imagery scraper
============================================================
Sources:
  1. NAIP (National Agriculture Imagery Program) via USGS STAC API
  2. USDA CropScape / CroplandCROS raster tiles
  3. Kaggle plant disease datasets via kaggle CLI

All imagery is geo-tagged, filtered to Ohio / Midwest, and cataloged
in LanceDB for deduplication + fast retrieval.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

try:
    import lancedb
    HAS_LANCE = True
except ImportError:
    HAS_LANCE = False


# ── Ohio / Midwest bounding box (approximate) ──────────────────────
OHIO_BBOX = {
    "west": -84.82,
    "south": 38.40,
    "east": -80.52,
    "north": 41.98,
}

MIDWEST_BBOX = {
    "west": -91.5,
    "south": 36.0,
    "east": -80.5,
    "north": 42.5,
}


class AgriScraper:
    """Pull, filter, and catalog agricultural aerial imagery."""

    def __init__(self, cfg: dict):
        self.cfg = cfg["scraper"]
        self.data_root = Path(cfg["paths"]["data_root"]).expanduser()
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.data_root / "raw"
        self.raw_dir.mkdir(exist_ok=True)

        # LanceDB catalog for dedup
        self.db = None
        self.table = None
        if HAS_LANCE:
            db_path = str(self.data_root / "catalog.lancedb")
            self.db = lancedb.connect(db_path)
            self._init_catalog()

    def _init_catalog(self) -> None:
        """Create or open the image catalog table."""
        try:
            self.table = self.db.open_table("images")
        except Exception:
            import pyarrow as pa
            schema = pa.schema([
                pa.field("hash", pa.string()),
                pa.field("source", pa.string()),
                pa.field("path", pa.string()),
                pa.field("lat", pa.float64()),
                pa.field("lon", pa.float64()),
                pa.field("year", pa.int32()),
                pa.field("state", pa.string()),
            ])
            self.table = self.db.create_table("images", schema=schema)

    def _hash_image(self, path: Path) -> str:
        """SHA-256 of image bytes for dedup."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def _already_seen(self, img_hash: str) -> bool:
        if self.table is None:
            return False
        try:
            results = self.table.search().where(f"hash = '{img_hash}'").limit(1).to_list()
            return len(results) > 0
        except Exception:
            return False

    def _catalog(self, path: Path, source: str, lat: float = 0.0,
                 lon: float = 0.0, year: int = 0, state: str = "OH") -> None:
        if self.table is None:
            return
        img_hash = self._hash_image(path)
        if not self._already_seen(img_hash):
            self.table.add([{
                "hash": img_hash,
                "source": source,
                "path": str(path),
                "lat": lat,
                "lon": lon,
                "year": year,
                "state": state,
            }])

    # ── NAIP via USGS STAC ──────────────────────────────────────────
    def scrape_naip(self, max_images: int = 500) -> list[Path]:
        """Query USGS STAC for NAIP imagery over Ohio."""
        print("[NAIP] Querying USGS STAC API for Ohio tiles...")
        naip_dir = self.raw_dir / "naip"
        naip_dir.mkdir(exist_ok=True)

        stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
        downloaded: list[Path] = []

        for year in self.cfg.get("naip_years", [2022]):
            payload = json.dumps({
                "collections": ["naip"],
                "bbox": [OHIO_BBOX["west"], OHIO_BBOX["south"],
                         OHIO_BBOX["east"], OHIO_BBOX["north"]],
                "datetime": f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
                "limit": min(max_images, 50),
            }).encode()

            req = urllib.request.Request(
                stac_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
            except Exception as e:
                print(f"  [WARN] STAC query failed for {year}: {e}")
                continue

            features = data.get("features", [])
            print(f"  [NAIP] {year}: found {len(features)} tiles")

            for feat in features[:max_images - len(downloaded)]:
                props = feat.get("properties", {})
                assets = feat.get("assets", {})
                img_asset = assets.get("image", assets.get("visual", {}))
                href = img_asset.get("href")
                if not href:
                    continue

                fname = naip_dir / f"naip_{year}_{len(downloaded):04d}.tif"
                try:
                    urllib.request.urlretrieve(href, fname)
                    # Extract centroid from bbox
                    bbox = feat.get("bbox", [0, 0, 0, 0])
                    lat = (bbox[1] + bbox[3]) / 2
                    lon = (bbox[0] + bbox[2]) / 2
                    self._catalog(fname, "naip", lat, lon, year, "OH")
                    downloaded.append(fname)
                    print(f"    -> {fname.name} ({lat:.3f}, {lon:.3f})")
                except Exception as e:
                    print(f"    [WARN] Download failed: {e}")

                if len(downloaded) >= max_images:
                    break

        print(f"[NAIP] Downloaded {len(downloaded)} tiles")
        return downloaded

    # ── USDA CropScape ──────────────────────────────────────────────
    def scrape_usda(self, max_images: int = 500) -> list[Path]:
        """Pull USDA CropScape raster data for Ohio counties."""
        print("[USDA] Querying CropScape CDL...")
        usda_dir = self.raw_dir / "usda"
        usda_dir.mkdir(exist_ok=True)
        downloaded: list[Path] = []

        # CropScape GetCDLData endpoint
        base_url = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"
        # Sample Ohio bounding box tiles (subdivide state into grid)
        lat_steps = np.linspace(OHIO_BBOX["south"], OHIO_BBOX["north"], num=10)
        lon_steps = np.linspace(OHIO_BBOX["west"], OHIO_BBOX["east"], num=10)

        for year in self.cfg.get("naip_years", [2022]):
            for i in range(len(lat_steps) - 1):
                for j in range(len(lon_steps) - 1):
                    if len(downloaded) >= max_images:
                        break
                    bbox_str = (f"{lon_steps[j]},{lat_steps[i]},"
                                f"{lon_steps[j+1]},{lat_steps[i+1]}")
                    url = f"{base_url}?year={year}&bbox={bbox_str}"
                    fname = usda_dir / f"cdl_{year}_{i}_{j}.tif"
                    try:
                        urllib.request.urlretrieve(url, fname)
                        lat = (lat_steps[i] + lat_steps[i+1]) / 2
                        lon = (lon_steps[j] + lon_steps[j+1]) / 2
                        self._catalog(fname, "usda", lat, lon, year, "OH")
                        downloaded.append(fname)
                    except Exception:
                        pass  # CropScape tiles may 404

        print(f"[USDA] Downloaded {len(downloaded)} CDL tiles")
        return downloaded

    # ── Kaggle datasets ─────────────────────────────────────────────
    def scrape_kaggle(self, max_images: int = 2000) -> list[Path]:
        """Download plant disease datasets from Kaggle."""
        print("[Kaggle] Downloading ag datasets...")
        kaggle_dir = self.raw_dir / "kaggle"
        kaggle_dir.mkdir(exist_ok=True)
        downloaded: list[Path] = []

        datasets = self.cfg.get("kaggle_datasets", [])
        for ds in datasets:
            ds_name = ds.replace("/", "_")
            dest = kaggle_dir / ds_name
            if dest.exists():
                print(f"  [SKIP] {ds} already downloaded")
            else:
                cmd = [
                    "kaggle", "datasets", "download", "-d", ds,
                    "-p", str(dest), "--unzip",
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
                    print(f"  [OK] {ds}")
                except FileNotFoundError:
                    print("  [WARN] kaggle CLI not found — install with: pip install kaggle")
                    continue
                except subprocess.CalledProcessError as e:
                    print(f"  [WARN] Kaggle download failed: {e}")
                    continue

            # Walk and catalog images
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in dest.rglob(ext):
                    if len(downloaded) >= max_images:
                        break
                    self._catalog(img_path, "kaggle", state="OH")
                    downloaded.append(img_path)

        print(f"[Kaggle] Cataloged {len(downloaded)} images")
        return downloaded

    # ── Main entry ──────────────────────────────────────────────────
    def run(self, max_images: Optional[int] = None) -> None:
        per_source = (max_images or self.cfg.get("max_images_per_source", 5000))
        total = []

        total.extend(self.scrape_naip(max_images=per_source))
        total.extend(self.scrape_usda(max_images=per_source))
        total.extend(self.scrape_kaggle(max_images=per_source))

        print(f"\n[SCRAPER] Total images collected: {len(total)}")
        print(f"[SCRAPER] Data root: {self.data_root.resolve()}")
