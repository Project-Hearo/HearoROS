FROM yahboomtechnology/ros-humble:4.1.2
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash","-lc"]

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg lsb-release software-properties-common; \
    rm -rf /var/lib/apt/lists/*

RUN set -e; \
    rm -f /etc/apt/sources.list.d/ros2* || true; \
    rm -f /usr/share/keyrings/ros-archive-keyring.gpg || true; \
    . /etc/os-release; \
    export ROS_APT_SOURCE_VERSION="$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F tag_name | awk -F\" '{print $4}')"; \
    curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${VERSION_CODENAME}_all.deb"; \
    dpkg -i /tmp/ros2-apt-source.deb || true; \
    apt-get update; \
    apt-get -f install -y; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    add-apt-repository -y universe; \
    apt-get update --fix-missing; \
    apt-get install -y --no-install-recommends \
      build-essential cmake git python3-pip \
      python3-colcon-common-extensions python3-vcstool \
      python3-paho-mqtt; \
    rm -rf /var/lib/apt/lists/*


RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    python3-requests \
    python3-boto3 \
    # ros-humble-tf2-msgs \
    # ros-humble-sensor-msgs \
    # ros-humble-nav-msgs \
    ros-humble-slam-toolbox \
    ros-humble-rviz2 \
    ros-humble-nav2-map-server \
    ros-humble-nav2-bringup \
    ros-humble-nav2-costmap-2d \
    ros-humble-nav2-amcl \
    ros-humble-geometry-msgs \
    ros-humble-tf2 \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
 && rm -rf /var/lib/apt/lists/*



WORKDIR /root/colcon_ws
RUN mkdir -p src

ARG GIT_REFRESH=initial

RUN echo "GIT_REFRESH=${GIT_REFRESH}"
RUN if [ ! -d src/m-explore-ros2 ]; then \
      git clone --depth=1 https://github.com/robo-friends/m-explore-ros2.git src/m-explore-ros2; \
    else \
      git -C src/m-explore-ros2 fetch --depth=1 origin && git -C src/m-explore-ros2 reset --hard origin/HEAD; \
    fi

RUN if [ ! -d src/room2d_sim ]; then \
      source /opt/ros/humble/setup.bash; \
      ros2 pkg create room2d_sim \
        --build-type ament_python \
        --dependencies rclpy sensor_msgs nav_msgs geometry_msgs tf2_ros tf_transformations; \
    fi

RUN set -e; \
    source /opt/ros/humble/setup.bash; \
    rosdep update; \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro humble

RUN set -e; \
    source /opt/ros/humble/setup.bash; \
    colcon build --symlink-install --event-handlers console_direct+


RUN printf '%s\n' \
  'source /opt/ros/humble/setup.bash' \
  'if [ -f /root/colcon_ws/install/setup.bash ]; then source /root/colcon_ws/install/setup.bash; fi' \
  'export QT_X11_NO_MITSHM=1' \
  'exec "$@"' > /ros_entry.sh && chmod +x /ros_entry.sh

ENTRYPOINT ["/bin/bash", "/ros_entry.sh"]
CMD ["/bin/bash"]
