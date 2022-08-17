#!/bin/bash

docker run \
	--rm \
	--network host \
	-v $PWD/code:/code \
       	-e DISPLAY=$DISPLAY \
       	-v /tmp/.X11-unix:/tmp/.X11-unix \
        -it crowd-analysis /code/run.sh

