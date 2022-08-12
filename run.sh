#!/bin/bash

docker run \
	--rm \
	--network host \
	-v $PWD/code:/code \
        -it crowd-analysis /code/run.sh

