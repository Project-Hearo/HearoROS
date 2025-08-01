docker rm -f ros_container

xhost +
docker run -it --rm\
  --privileged=true \
  --net=host \
  -e DISPLAY=$DISPLAY \
  --env="QT_X11_NO_MITSHM=1" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --security-opt apparmor:unconfined \
  -v /dev/input:/dev/input \
  -v /dev/video0:/dev/video0 \
  -v /var/run/dbus:/var/run/dbus \
  -v ~/Desktop/HearoROS:/root/HearoROS \
  yahboomtechnology/ros-humble:4.1.2 \
  /bin/bash \
  /root/1.sh


  # 일부 QT GUI앱이 공유 메모리 미사용으로 돌아가게 설정
# GUI출력을 위한 X11 소켓 공유
  # 장치 입출력 접근 기능
  # 키보드, 마우스, USB카메라 등등
#컨테이너 안에서 bash /root/1.sh가 실행된다.
# 호스트의 프로젝트 폴더를 컨테이너의 /root/HearoROS로 마운트