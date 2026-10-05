#!/usr/bin/env bash
# Start the Khronos container with the host repo mounted, so files can be edited from the host.
# Usage: ./docker_run.sh [path/to/rosbags]   (default: ~/rosbags, mounted at /data)
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAG_DIR="$(realpath -m "${1:-$HOME/rosbags}")"
IMAGE="${KHRONOS_IMAGE:-khronos:jazzy}"

mkdir -p "$BAG_DIR"
xhost +local:docker >/dev/null 2>&1 || echo "warning: xhost failed, GUI windows may not open"

GPU_ARGS=()
if command -v nvidia-ctk >/dev/null 2>&1; then GPU_ARGS=(--gpus all); fi

docker run -it --rm --net=host \
  -e DISPLAY="$DISPLAY" -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --device /dev/dri \
  "${GPU_ARGS[@]}" \
  -v "$BAG_DIR":/data \
  -v "$REPO_DIR":/root/ros2_ws/src/khronos \
  "$IMAGE"
