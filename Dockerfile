FROM ubuntu:18.04
COPY . /
RUN export DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -yq python3 python3-pip virtualenv
RUN virtualenv --python=$(which python3) venv
RUN source /venv/bin/activate
RUN pip install -r requirements.txt
CMD python zwee.py
