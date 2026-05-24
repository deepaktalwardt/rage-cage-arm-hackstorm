FROM piper:jazzy

# `python3 -m pip` (not `pip3`) installs into whatever Python the
# container's `python3` resolves to. On this base, `pip3` targets a venv
# at /opt/venv that the system `python3` (/usr/local/bin/python3) does
# NOT have in its sys.path — so packages installed via `pip3` are
# invisible to our node.
#
# --index-url overrides the base image's default (pypi.jetson-ai-lab.dev),
# which isn't reachable in our network setup.
#
# --ignore-installed bypasses pip's attempt to uninstall the Debian-
# installed numpy 1.26 (no RECORD file, can't be uninstalled). Pip writes
# numpy 2.x to /usr/local/lib/python3.12/dist-packages alongside the
# system 1.26 copy.
COPY real/docker/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --break-system-packages --ignore-installed \
      --index-url https://pypi.org/simple \
      -r /tmp/requirements.txt

# Clone + colcon-build the agx_arm_ros workspace. Provides the custom
# message types (agx_arm_msgs/MoveMITMsg, AgxArmStatus, ...) and the
# agx_arm_ctrl driver node (start_single_agx_arm.launch.py). The base
# piper:jazzy image pip-installs pyAgxArm (the Python SDK) but doesn't
# build the ROS-side packages, so `from agx_arm_msgs.msg import ...`
# fails until we do this.
#
# --merge-install lands all packages under one /opt/agx_arm_ws/install/
# prefix so the PYTHONPATH addition stays a single entry.
# --recurse-submodules pulls in the agx_arm_urdf submodule referenced
# by agx_arm_description.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/agx_arm_ws/src \
 && git clone -b ros2 --recurse-submodules \
        https://github.com/agilexrobotics/agx_arm_ros.git \
        /opt/agx_arm_ws/src/agx_arm_ros \
 && bash -c "source /opt/ros/jazzy/install/setup.bash \
          && cd /opt/agx_arm_ws \
          && colcon build --merge-install --symlink-install \
                --cmake-args -DBUILD_TESTING=OFF"

# Path setup, in priority order:
#   /usr/local/lib/python3.12/dist-packages
#     numpy 2.x install dir, FIRST so it shadows the Debian numpy 1.26
#     at /usr/lib/python3/dist-packages. vecnormalize.pkl was pickled
#     with numpy 2.x and references the `numpy._core` private path that
#     doesn't exist in 1.x — without this shadow, unpickle fails with
#     ModuleNotFoundError.
#   /opt/agx_arm_ws/install/lib/python3.12/site-packages
#     agx_arm_msgs + agx_arm_ctrl Python bindings from the colcon build
#     above.
#   ${PYTHONPATH}
#     Everything inherited from piper:jazzy (rclpy, std_msgs,
#     sensor_msgs, etc.).
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/opt/agx_arm_ws/install/lib/python3.12/site-packages:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/agx_arm_ws/install/lib:${LD_LIBRARY_PATH}

# Source the agx_arm_ws install in interactive shells so things like
# `ros2 topic echo /control/move_mit` (and the `agx_arm_ctrl` launch
# files) Just Work without manual sourcing every time.
RUN echo '[ -f /opt/agx_arm_ws/install/setup.bash ] && source /opt/agx_arm_ws/install/setup.bash' \
        >> /root/.bashrc

# Code and models come in via volume mount; see compose.yaml.
# WORKDIR /ws is inherited from the base image.
