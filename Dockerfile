# Khronos (ROS 2) image: Ubuntu 24.04 + ROS 2 Jazzy, following the README installation steps.
#
# Build (from the repository root):
#   docker build -t khronos:jazzy .
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_WS=/root/ros2_ws

# README: "Setup a ROS2 workspace" + "Install system dependencies", plus rviz and GL support.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        python3-vcstool python3-tk python3-colcon-common-extensions python3-pip python3-venv \
        ros-jazzy-gtsam libgoogle-glog-dev nlohmann-json3-dev \
        ros-jazzy-rviz2 ros-jazzy-rosbag2-storage-mcap ros-jazzy-image-transport-plugins \
        libgl1 libglx-mesa0 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*

# README: "Get Khronos and all source dependencies" (https variant, no ssh key needed).
COPY . ${ROS_WS}/src/khronos
WORKDIR ${ROS_WS}/src
RUN vcs import . < khronos/install/https.rosinstall

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
