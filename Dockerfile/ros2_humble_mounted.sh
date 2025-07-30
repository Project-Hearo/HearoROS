docker rm -f ros_container

docker run -it --rm\
  --privileged=true \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/input:/dev/input \
  -v /dev/video0:/dev/video0 \
  -v ~/Desktop/HearoROS:/root/HearoROS \
  yahboomtechnology/ros-humble:4.1.2 \
  /bin/bash
