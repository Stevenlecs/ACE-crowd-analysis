ARG IMAGE=tensorflow/tensorflow
ARG VERSION=latest-gpu
FROM $IMAGE:$VERSION

# Install opencv
RUN apt update \
        && apt upgrade -y \
        && apt install -y \
		git \
		libglib2.0-0 \
                libgl1-mesa-glx
RUN pip install opencv-python

# Install Crowd 
RUN apt install -y wget
#RUN git clone https://github.com/lewjiayi/Crowd-Analysis.git /code/Crowd-Analysis \
#	&& cd /code/Crowd-Analysis \
#	&& mkdir YOLOv4-tiny \
#	&& wget -P YOLOv4-tiny https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights \ 
#	&& wget -P YOLOv4-tiny https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg 

RUN /usr/bin/python3 -m pip install --upgrade pip
COPY ./requirements.txt /code/requirements.txt
RUN cd /code && pip install -r requirements.txt

# Install ACE
RUN git clone https://github.com/usnistgov/ACE.git /opt/ace \
	&& cd /opt/ace \
	&& pip install .

WORKDIR /code

