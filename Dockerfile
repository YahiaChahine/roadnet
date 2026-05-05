FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install --upgrade pip

# Clone CityFlow fresh with submodules instead of copying
RUN git clone --recursive https://github.com/cityflow-project/CityFlow.git /tmp/CityFlow && \
    sed -i 's/cmake_minimum_required(VERSION 2.8)/cmake_minimum_required(VERSION 3.5)/' /tmp/CityFlow/CMakeLists.txt && \
    cd /tmp/CityFlow && pip install . && \
    rm -rf /tmp/CityFlow

RUN pip install tensorflow-gpu==2.4.0 pandas numpy

WORKDIR /workspace/code
