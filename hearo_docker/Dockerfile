FROM yahboomtechnology/ros-humble:4.1.2
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-slam-toolbox \
    ros-humble-rviz2 \
    ros-humble-nav2-map-server \
    ros-humble-nav2-bringup \
    ros-humble-nav2-costmap-2d \
    ros-humble-amcl \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/colcon_ws/src
RUN . /opt/ros/humble/setup.sh && \
    rosdep update && \
    rosdep install --from-paths /root/colcon_ws/src --ignore-src -r -y || true && \
    colcon build --symlink-install || true


RUN printf '%s\n' \
  'source /opt/ros/humble/setup.bash' \
  'if [ -f /root/colcon_ws/install/setup.bash ]; then source /root/colcon_ws/install/setup.bash; fi' \
  'export QT_X11_NO_MITSHM=1' \
  'exec "$@"' > /ros_entry.sh && chmod +x /ros_entry.sh

ENTRYPOINT ["/ros_entry.sh"]
CMD ["/bin/bash"]
