#!/usr/bin/env bash
# Start the Khronos container with the host repo mounted, so files can be edited from the host.
# Usage: ./docker_run.sh [path/to/datasets]   (mounted at /data)
# Default datasets folder: $KHRONOS_DATASETS, else <workspace>/datasets if the repo sits in <workspace>/src/khronos, else ~/rosbags.
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_BAG_DIR="${KHRONOS_DATASETS:-$REPO_DIR/../../datasets}"
[ -d "$DEFAULT_BAG_DIR" ] || DEFAULT_BAG_DIR="$HOME/rosbags"
BAG_DIR="$(realpath -m "${1:-$DEFAULT_BAG_DIR}")"
IMAGE="${KHRONOS_IMAGE:-khronos:jazzy}"

mkdir -p "$BAG_DIR"
xhost +local:docker >/dev/null 2>&1 || echo "warning: xhost failed, GUI windows may not open"

# Mount a sibling dynamic_psg checkout over the image's copy, so its changes rebuild too.
DYNAMIC_PSG_ARGS=()
DYNAMIC_PSG_DIR="$(realpath -m "$REPO_DIR/../dynamic_psg")"
if [ -d "$DYNAMIC_PSG_DIR" ]; then
  DYNAMIC_PSG_ARGS=(-v "$DYNAMIC_PSG_DIR":/root/ros2_ws/src/dynamic_psg)
fi

GPU_ARGS=()
if command -v nvidia-ctk >/dev/null 2>&1; then GPU_ARGS=(--gpus all); fi

docker run -it --rm --net=host \
  -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri \
  "${GPU_ARGS[@]}" \
  -v "$BAG_DIR":/data \
  -v "$REPO_DIR":/root/ros2_ws/src/khronos \
  "${DYNAMIC_PSG_ARGS[@]}" \
  "$IMAGE"
