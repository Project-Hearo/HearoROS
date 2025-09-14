set -euo pipefail

docker build -t hearo/ros-humble-slam:latest .

xhost +local:root

CLEAN_BUILD_AND_SHELL='
set -e
[ -f /opt/ros/humble/setup.bash ] && source /opt/ros/humble/setup.bash

WS="/root/HearoROS"
if [ ! -d "$WS" ] || [ ! -d "$WS/src" ]; then
  echo "src가 없습니다"
  exec bash -l
fi

echo "build/install/log 제거"
rm -rf "$WS/build" "$WS/install" "$WS/log"

echo "rosdep (실패해도 계속 진행)"
set +e
rosdep update
rosdep install --from-paths "$WS/src" --ignore-src -r -y --rosdistro humble
set -e

echo "colcon build 시작"
cd "$WS"
colcon build --symlink-install --event-handlers console_direct+

echo "소싱"
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash" || true

echo "대화형 쉘 진입"
exec bash -l
'

docker run -it --rm \
  --name ros_container \
  --shm-size=1g \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --privileged \
  --security-opt apparmor:unconfined \
  -v /dev:/dev \
  --group-add video \
  --group-add render \
  -v ~/Desktop/HearoROS:/root/HearoROS \
  -v /var/run/dbus:/var/run/dbus \
  --env-file ~/Desktop/HearoROS/.env \
  -e ROS_DOMAIN_ID=20 \
  hearo/ros-humble-slam:latest \
  /bin/bash -lc "$CLEAN_BUILD_AND_SHELL"