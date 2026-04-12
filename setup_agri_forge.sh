#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# AGRI-FORGE — One-command setup
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo " AGRI-FORGE Setup"
echo "============================================================"

# ── 1. Python venv ──────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "[1/6] Creating Python virtual environment..."
    python3 -m venv .venv
else
    echo "[1/6] Virtual environment already exists"
fi

source .venv/bin/activate
echo "  Python: $(python3 --version)"

# ── 2. Install dependencies ────────────────────────────────────────
echo "[2/6] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q 2>&1 | tail -5
echo "  Dependencies installed"

# ── 3. Verify CUDA ─────────────────────────────────────────────────
echo "[3/6] Checking CUDA..."
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f'  GPU: {name}')
    print(f'  VRAM: {free/(1024**3):.1f} / {total/(1024**3):.1f} GB free')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  WARNING: No CUDA GPU detected')
"

# ── 4. Create directory structure ──────────────────────────────────
echo "[4/6] Creating directories..."
mkdir -p datasets/{raw/{naip,usda,kaggle},synthetic/{images,labels}}
mkdir -p models
mkdir -p export/builds
mkdir -p reports/{swarm,validation}
echo "  Directory tree ready"

# ── 5. Download YOLOv8n base weights ──────────────────────────────
echo "[5/6] Downloading YOLOv8n base weights..."
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('  YOLOv8n weights ready')
" 2>/dev/null || echo "  WARNING: Could not download YOLOv8n (will retry on first train)"

# ── 6. ROS2 Humble/Jazzy setup for Isaac Sim bridge ────────────────
echo "[6/8] Checking ROS2 for Isaac Sim bridge..."
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    echo "  ROS2 Humble: sourced"
elif [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo "  ROS2 Jazzy: sourced"
else
    echo "  ROS2 not found. To install ROS2 Humble for Isaac Sim bridge:"
    echo "    sudo apt update && sudo apt install -y software-properties-common"
    echo "    sudo add-apt-repository universe"
    echo "    sudo apt update && sudo apt install curl -y"
    echo '    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg'
    echo '    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null'
    echo "    sudo apt update && sudo apt install -y ros-humble-desktop ros-humble-rmw-fastrtps-cpp"
    echo "  (Isaac Sim procedural fallback will still work without ROS2)"
fi

# ── 7. Isaac Sim WSL bridge environment ────────────────────────────
echo "[7/8] Configuring Isaac Sim WSL bridge..."
ISAAC_WIN_PATH="C:\\isaacsim"
ISAAC_WSL_PATH="/mnt/c/isaacsim"
if [ -d "$ISAAC_WSL_PATH" ]; then
    echo "  Isaac Sim found at $ISAAC_WSL_PATH"
    # Create shared data directory on Windows side
    SHARED_DIR="/mnt/c/agri_forge_isaac_data"
    mkdir -p "$SHARED_DIR"/{usd_scenes,renders,labels,depth,segmentation} 2>/dev/null || true
    echo "  Shared dir: $SHARED_DIR"
else
    echo "  Isaac Sim NOT found at $ISAAC_WSL_PATH"
    echo "  Install Isaac Sim on Windows at $ISAAC_WIN_PATH for photorealistic rendering"
    echo "  (Procedural mode with --mode synthetic will still work)"
fi

# Export env vars for ROS2 ↔ Isaac DDS communication
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE=""  # will be set at runtime

# ── 8. Verify Cosmos-Reason2 cache ────────────────────────────────
echo "[8/8] Checking Cosmos-Reason2 cache..."
COSMOS_CACHE="${HOME}/.cache/huggingface/hub"
if ls "$COSMOS_CACHE"/models--nvidia--Cosmos-Reason2-2B* 1>/dev/null 2>&1; then
    echo "  Cosmos-Reason2-2B: cached"
else
    echo "  Cosmos-Reason2-2B: not cached (will download on first use)"
    echo "  To pre-download: python3 -c \"from transformers import AutoModel; AutoModel.from_pretrained('nvidia/Cosmos-Reason2-2B', trust_remote_code=True)\""
fi

# ── 9. Verify test dependencies work in clean env ──────────────────
echo "[EXTRA] Verifying pytest runs clean (isolating ROS2 paths)..."
(
    # Run pytest in a subshell with ROS paths stripped to avoid lark conflicts
    unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PYTHON_VERSION
    # Only keep non-ROS entries in PYTHONPATH
    export PYTHONPATH=$(python3 -c "
import os, sys
paths = os.environ.get('PYTHONPATH', '').split(':')
clean = [p for p in paths if '/opt/ros' not in p]
print(':'.join(clean))
")
    python3 -m pytest tests/ -q --tb=no 2>&1 | tail -3
) || echo "  (some tests may require GPU or data — run manually after setup)"

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " AGRI-FORGE Setup Complete!"
echo "============================================================"
echo ""
echo " Activate:  source .venv/bin/activate"
echo ""
echo " Quick start (procedural):"
echo "   python agri_forge.py --mode synthetic --num-images 100"
echo ""
echo " Quick start (Isaac Sim photorealistic):"
echo "   python agri_forge.py --mode isaac --num-images 10 --headless"
echo "   python agri_forge.py --mode isaac --num-images 2000 --headless"
echo ""
echo " Other modes:"
echo "   python agri_forge.py --mode swarm --num-agents 20 --duration 5"
echo "   python agri_forge.py --mode train --config config.yaml"
echo "   python agri_forge.py --mode metrics"
echo ""
echo " Run tests: python -m pytest tests/ -v"
echo "============================================================"
