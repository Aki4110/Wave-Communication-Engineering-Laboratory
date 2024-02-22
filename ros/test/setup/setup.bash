docker run -it --rm --privileged \
    -e DISPLAY=$DISPLAY \
    -e XDG_RUNTIME_DIR=/tmp/runtime-root \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /home/sakuramoto/home_workspace/ros/test/workspace:/app/workspace \
    ros-test

