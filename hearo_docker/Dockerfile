FROM yahboomtechnology/ros-humble:4.1.2
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash","-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg lsb-release && \
    rm -rf /var/lib/apt/lists/*

RUN set -e; \
    rm -f /etc/apt/sources.list.d/ros2* || true; \
    rm -f /usr/share/keyrings/ros-archive-keyring.gpg || true; \
    . /etc/os-release; \
    curl -fsSL -o /tmp/ros2-apt-source.deb \
      "https://github.com/ros-infrastructure/ros-apt-source/releases/latest/download/ros2-apt-source_${VERSION_CODENAME}_all.deb"; \
    dpkg -i /tmp/ros2-apt-source.deb || true; \
    apt-get update; \
    apt-get -f install -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-slam-toolbox \
    ros-humble-rviz2 \
    ros-humble-nav2-map-server \
    ros-humble-nav2-bringup \
    ros-humble-nav2-costmap-2d \
    ros-humble-nav2-amcl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root/colcon_ws
RUN mkdir -p src

RUN source /opt/ros/humble/setup.bash && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y || true && \
    colcon build --symlink-install || true

RUN printf '%s\n' \
  'source /opt/ros/humble/setup.bash' \
  'if [ -f /root/colcon_ws/install/setup.bash ]; then source /root/colcon_ws/install/setup.bash; fi' \
  'export QT_X11_NO_MITSHM=1' \
  'exec "$@"' > /ros_entry.sh && chmod +x /ros_entry.sh

ENTRYPOINT ["/ros_entry.sh"]
CMD ["/bin/bash"]
