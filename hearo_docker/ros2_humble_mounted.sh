docker build -t hearo/ros-humble-slam:latest .

xhost +local:root

docker run -it --rm \
  --name ros_container \
  --privileged \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e ROS_DOMAIN_ID=20 \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --security-opt apparmor:unconfined \
  -v /dev/input:/dev/input \
  -v /dev/video0:/dev/video0 \
  -v /var/run/dbus:/var/run/dbus \
  -v /dev/dri:/dev/dri \        
  -v ~/Desktop/HearoROS:/root/HearoROS \
  hearo/ros-humble-slam:latest \
  /bin/bash /root/1.sh          

