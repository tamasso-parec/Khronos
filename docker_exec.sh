#!/usr/bin/env bash
# Open another terminal in an existing container; never creates a new one.
# Usage: ./docker_exec.sh [container-name-or-id]
set -euo pipefail

if [[ $# -gt 1 || ${1:-} == --help || ${1:-} == -h ]]; then
  echo "Usage: $0 [container-name-or-id]"
  exit 0
fi

CONTAINER="${1:-}"
if [[ -z "$CONTAINER" ]]; then
  RUNNING=$(docker ps --filter "ancestor=${KHRONOS_IMAGE:-khronos:jazzy}" --format '{{.Names}}')
  if [[ -z "$RUNNING" ]]; then
    echo "No running Khronos container found. Start one with ./docker_run.sh first." >&2
    exit 1
  fi
  mapfile -t CONTAINERS <<< "$RUNNING"
  if [[ ${#CONTAINERS[@]} -eq 1 ]]; then
    CONTAINER="${CONTAINERS[0]}"
  else
    echo "Choose the container running Khronos:"
    PS3="Container number: "
    select CONTAINER in "${CONTAINERS[@]}"; do
      [[ -n "$CONTAINER" ]] && break
      echo "Select a number from the list." >&2
    done
    [[ -n "$CONTAINER" ]] || exit 1
  fi
fi

echo "Opening a shell in $CONTAINER"
exec docker exec -it "$CONTAINER" bash -c '
  source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
  cd "${ROS_WS:-/root/ros2_ws}" || exit 1
  if [[ -f install/setup.bash ]]; then
    source install/setup.bash
  fi
  exec bash -i
'
