# syntax=docker/dockerfile:1
# Khronos (ROS 2) image: Ubuntu 24.04 + ROS 2 Jazzy, following the README installation steps.
#
# Build (from the repository root). dynamic_psg is private, so the build clones it over SSH
# with your agent's key (check with ssh-add -l):
#   docker build --ssh default -t khronos:jazzy .
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_WS=/root/ros2_ws

# README: "Setup a ROS2 workspace" + "Install system dependencies", plus rviz and GL support.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git openssh-client \
        python3-vcstool python3-tk python3-colcon-common-extensions python3-pip python3-venv \
        ros-jazzy-gtsam libgoogle-glog-dev nlohmann-json3-dev \
        ros-jazzy-rviz2 ros-jazzy-rosbag2-storage-mcap ros-jazzy-image-transport-plugins \
        libgl1 libglx-mesa0 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# README: "Get Khronos and all source dependencies". The private dynamic_psg repository is
# cloned over SSH first; the public dependencies then come over https.
COPY . ${ROS_WS}/src/khronos
WORKDIR ${ROS_WS}/src
RUN --mount=type=ssh mkdir -p -m 0700 /root/.ssh \
    && ssh-keyscan github.com >> /root/.ssh/known_hosts 2>/dev/null \
    && git clone git@github.com:utn-mi/dynamic_psg.git dynamic_psg
RUN vcs import --skip-existing . < khronos/install/https.rosinstall

# Remaining dependencies of all workspace packages.
RUN apt-get update \
    && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths . --ignore-src -r -y --rosdistro jazzy --skip-keys "teaserpp gtsam" \
    && rm -rf /var/lib/apt/lists/*

# README: "Build".
WORKDIR ${ROS_WS}
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc \
    && echo "source ${ROS_WS}/install/setup.bash" >> /root/.bashrc

CMD ["bash"]
