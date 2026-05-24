FROM docker.io/dustynv/ros:jazzy-desktop-r36.4.0-cu128-24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy

# Refresh the ROS 2 apt signing key + point at noble (24.04) repo
RUN apt-get update --allow-insecure-repositories \
        -o Acquire::AllowInsecureRepositories=true \
        -o Acquire::AllowDowngradeToInsecureRepositories=true 2>/dev/null || true && \
    apt-get install -y --allow-unauthenticated curl gnupg ca-certificates && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    rm -f /etc/apt/sources.list.d/ros2.list && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y --no-install-recommends \
      -o Dpkg::Options::="--force-overwrite" \
      can-utils iproute2 ethtool \
      python3-pip python3-colcon-common-extensions python3-rosdep \
      ros-jazzy-ros2-control ros-jazzy-ros2-controllers \
      ros-jazzy-controller-manager \
      ros-jazzy-xacro \
      ros-jazzy-joint-state-publisher \
      ros-jazzy-robot-state-publisher \
      ros-jazzy-rmw-cyclonedds-cpp \
      ros-jazzy-foxglove-bridge \
      ros-jazzy-realsense2-camera \
      ros-jazzy-tf2-tools \
      ros-jazzy-ros2launch \
      ros-jazzy-rosidl-default-generators \
      ros-jazzy-ament-lint-auto \
      ros-jazzy-ament-lint-common \
    && rm -rf /var/lib/apt/lists/*

# Noble enforces PEP 668; --break-system-packages is the container-appropriate escape hatch.
# PIP_INDEX_URL is set as env (not just --index-url) so pip's build isolation child
# also reaches pypi.org — the container's default index (pypi.jetson-ai-lab.dev)
# can't resolve from inside the build context, which breaks build deps for git installs.
RUN PIP_INDEX_URL=https://pypi.org/simple \
    pip3 install --no-cache-dir --break-system-packages \
      --index-url https://pypi.org/simple \
      "opencv-python-headless>=4.10.0" \
      "python-can>=4.3.1" scipy ikpy \
      "pyAgxArm @ git+https://github.com/agilexrobotics/pyAgxArm.git"

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Bake ROS environment into the image so `docker exec piper_ros ros2 ...`
# works without needing to source setup.bash. Values mirror what sourcing
# /opt/ros/jazzy/install/setup.bash + /opt/ros/jazzy/setup.bash produces.
# /usr/local/bin is put first on PATH so the workspace-overlay wrapper below
# is found before the bare ros2 binary.
ENV AMENT_PREFIX_PATH=/opt/ros/jazzy:/opt/ros/jazzy/install \
    PATH=/usr/local/bin:/opt/ros/jazzy/install/bin:/opt/ros/jazzy/bin:/opt/venv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/ros/jazzy/install/lib/python3.12/site-packages:/opt/ros/jazzy/lib/python3.12/site-packages:/usr/lib/python3/dist-packages:/opt/venv/lib/python3.12/site-packages \
    LD_LIBRARY_PATH=/opt/ros/jazzy/install/lib:/opt/ros/jazzy/lib/aarch64-linux-gnu:/opt/ros/jazzy/lib:/usr/local/cuda/lib64 \
    ROS_VERSION=2

# Wrapper at /usr/local/bin/ros2 sources the workspace overlay (if built)
# before exec'ing the real ros2. Needed so custom message types like
# piper_msgs/PiperStatusMsg are deserialisable in `ros2 topic echo`.
RUN printf '%s\n' \
        '#!/bin/bash' \
        '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' \
        'exec /opt/ros/jazzy/install/bin/ros2 "$@"' \
        > /usr/local/bin/ros2 \
    && chmod +x /usr/local/bin/ros2

RUN { \
      echo "source /opt/ros/jazzy/install/setup.bash"; \
      echo "source /opt/ros/jazzy/setup.bash"; \
    } >> /root/.bashrc

WORKDIR /ws
CMD ["sleep", "infinity"]
