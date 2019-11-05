# Minimal Surreal Image for passing all travis test cases.

FROM tensorflow/tensorflow:2.0.0-py3

RUN apt-get update -y
RUN apt-get upgrade -y

# Needed for opencv-python.
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Upgrade pip before installing packages with it.
RUN pip3 install --upgrade pip
# Install necessary pip packages.
RUN pip3 install numpy pydevd tensorflow tensorflow_probability gym scipy pyyaml gym[atari] opencv-python pytest \
    requests lz4 pyarrow wheel atari_py pygame
#RUN pip3 install unittest

CMD ["bash"]
