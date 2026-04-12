#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Florio Industries — Photogrammetry Pipeline
# ═══════════════════════════════════════════════════════════════
# Processes geotagged drone imagery into:
#   1. Orthomosaic (2D geo-referenced map)
#   2. Digital Surface Model (elevation/topography)
#   3. 3D textured mesh (digital twin)
#   4. 3D tiles (web-viewable model)
#   5. Point cloud (LAS format)
#
# Runs on your WORKSTATION PC, not the Jetson.
# Transfer images from Jetson via USB drive after mission.
#
# Prerequisites:
#   Docker:  docker pull opendronemap/odm
#   Or native: pip install OpenDroneMap (requires many deps)
#
# Usage:
#   ./photogrammetry_pipeline.sh /path/to/mission_images [output_dir]
#
# The input directory should contain geotagged JPEGs from the
# mission logger (~/firmament_data/ag_detections/) or from
# a dedicated photogrammetry flight.
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Arguments ───────────────────────────────────────────────────
INPUT_DIR="${1:?Usage: $0 <input_images_dir> [output_dir]}"
OUTPUT_DIR="${2:-${INPUT_DIR}_processed}"
MISSION_NAME=$(basename "$INPUT_DIR")

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  FLORIO INDUSTRIES — Photogrammetry Pipeline"
echo "═══════════════════════════════════════════════════════════"
echo "  Input:   $INPUT_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "  Mission: $MISSION_NAME"
echo "═══════════════════════════════════════════════════════════"
echo ""

# ── Validate Input ──────────────────────────────────────────────
IMAGE_COUNT=$(find "$INPUT_DIR" -maxdepth 1 -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" | wc -l)
if [ "$IMAGE_COUNT" -lt 5 ]; then
    echo "ERROR: Found only $IMAGE_COUNT images. Need at least 5 for photogrammetry."
    echo "       (20+ recommended for good results)"
    exit 1
fi
echo "Found $IMAGE_COUNT images."

# Check for GPS EXIF data
if command -v exiftool &> /dev/null; then
    FIRST_IMAGE=$(find "$INPUT_DIR" -maxdepth 1 -name "*.jpg" | head -1)
    GPS_CHECK=$(exiftool -GPSLatitude "$FIRST_IMAGE" 2>/dev/null || echo "")
    if [ -z "$GPS_CHECK" ]; then
        echo "WARNING: No GPS EXIF data found in images."
        echo "         Results will lack georeferencing."
        echo "         For best results, ensure camera saves GPS to EXIF."
    else
        echo "GPS data confirmed in images."
    fi
fi

# ── Create Output Directory ─────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

# ── Run OpenDroneMap ────────────────────────────────────────────
# Using Docker for clean, reproducible environment.
# If you installed ODM natively, replace docker command with:
#   run_odm.sh --project-path "$OUTPUT_DIR" ...

if command -v docker &> /dev/null; then
    echo ""
    echo "Running OpenDroneMap via Docker..."
    echo "This may take 30-120 minutes depending on image count and PC specs."
    echo ""
    
    docker run --rm -it \
        -v "$INPUT_DIR":/datasets/code/images \
        -v "$OUTPUT_DIR":/datasets/code \
        opendronemap/odm \
        --project-path /datasets \
        --dsm \
        --dtm \
        --orthophoto-resolution 2.0 \
        --mesh-octree-depth 12 \
        --use-3dmesh \
        --pc-quality high \
        --auto-boundary \
        --cog \
        --tiles \
        --max-concurrency "$(nproc)" \
        --ignore-gsd \
        --feature-quality high \
        --depthmap-resolution 1000

elif command -v odm_app &> /dev/null || command -v run_odm.sh &> /dev/null; then
    echo "Running ODM natively..."
    # Adjust path for your ODM installation
    run_odm.sh \
        --project-path "$OUTPUT_DIR" \
        --dsm --dtm \
        --orthophoto-resolution 2.0 \
        --mesh-octree-depth 12 \
        --use-3dmesh \
        --pc-quality high \
        --cog --tiles \
        --max-concurrency "$(nproc)"
else
    echo "ERROR: Neither Docker nor ODM found."
    echo ""
    echo "Install Docker:  https://docs.docker.com/get-docker/"
    echo "Then pull ODM:   docker pull opendronemap/odm"
    echo ""
    echo "Or install ODM natively:"
    echo "  git clone https://github.com/OpenDroneMap/ODM.git"
    echo "  cd ODM && bash configure.sh install"
    exit 1
fi

# ── Post-Processing ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Processing complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""

# List outputs
echo "Generated outputs:"
if [ -f "$OUTPUT_DIR/code/odm_orthophoto/odm_orthophoto.tif" ]; then
    SIZE=$(du -h "$OUTPUT_DIR/code/odm_orthophoto/odm_orthophoto.tif" | cut -f1)
    echo "  ✓ Orthomosaic (2D map):     $SIZE"
    echo "    → $OUTPUT_DIR/code/odm_orthophoto/odm_orthophoto.tif"
fi

if [ -f "$OUTPUT_DIR/code/odm_dem/dsm.tif" ]; then
    echo "  ✓ Digital Surface Model:     (elevation/topography)"
    echo "    → $OUTPUT_DIR/code/odm_dem/dsm.tif"
fi

if [ -f "$OUTPUT_DIR/code/odm_dem/dtm.tif" ]; then
    echo "  ✓ Digital Terrain Model:     (bare ground elevation)"
    echo "    → $OUTPUT_DIR/code/odm_dem/dtm.tif"
fi

if [ -d "$OUTPUT_DIR/code/odm_texturing" ]; then
    echo "  ✓ 3D Textured Mesh:         (digital twin)"
    echo "    → $OUTPUT_DIR/code/odm_texturing/"
fi

if [ -d "$OUTPUT_DIR/code/3d_tiles" ]; then
    echo "  ✓ 3D Tiles:                 (web-viewable model)"
    echo "    → $OUTPUT_DIR/code/3d_tiles/"
fi

if [ -f "$OUTPUT_DIR/code/odm_georeferencing/odm_georeferenced_model.laz" ]; then
    echo "  ✓ Point Cloud (LAZ):        (3D point data)"
    echo "    → $OUTPUT_DIR/code/odm_georeferencing/odm_georeferenced_model.laz"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Next steps:"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  View orthomosaic:  QGIS (free) or drag .tif into Google Earth"
echo "  View 3D model:     Open 3d_tiles/ in Cesium or three.js viewer"
echo "  View point cloud:  CloudCompare (free)"
echo "  Use as basemap:    Copy orthophoto .tif to Jetson for offline"
echo "                     Leaflet map tiles in the farmer dashboard"
echo ""
echo "  To generate offline map tiles from the orthomosaic:"
echo "    gdal2tiles.py -z 13-20 -w none odm_orthophoto.tif tiles/"
echo "    # Then copy tiles/ to Jetson: ~/firmament-ag/static/tiles/"
echo ""
echo "  To extract elevation data for topographic contours:"
echo "    gdal_contour -a elevation -i 0.5 dsm.tif contours.gpkg"
echo ""
