ARG IMAGE=datamachines/tensorflow_opencv
ARG VERSION=2.9.1_4.5.5-20220530
FROM $IMAGE:$VERSION

RUN /usr/bin/python3 -m pip install --upgrade pip

RUN mkdir /code
COPY ./requirements_cto.txt /code/requirements.txt
RUN cd /code && pip3 install -r requirements.txt

# Install ACE
RUN git clone https://github.com/usnistgov/ACE.git /opt/ace \
	&& cd /opt/ace \
	&& pip3 install .

WORKDIR /code

