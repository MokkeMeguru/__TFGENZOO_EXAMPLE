FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN apt-add-repository ppa:fish-shell/release-3
RUN apt-get update
RUN apt-get install -y zsh tmux wget git
RUN mkdir /workspace
RUN pip install -U pip
ADD test.txt /tmp/
RUN pip install -r /tmp/test.txt
# RUN pip install git+https://github.com/MokkeMeguru/TFGENZOO.git@master#egg=TFGENZOO
WORKDIR /workspace
