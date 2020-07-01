FROM tensorflow/tensorflow:2.2.0-gpu
RUN apt-get update
RUN apt-get install -y zsh tmux wget git
RUN mkdir /workspace
RUN pip install -U pip
ADD requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt && \
    pip install matplotlib
# RUN pip install git+https://github.com/MokkeMeguru/TFGENZOO.git@v1.2.1#egg=TFGENZOO
RUN pip install TFGENZOO==1.2.4.post6
WORKDIR /workspace
