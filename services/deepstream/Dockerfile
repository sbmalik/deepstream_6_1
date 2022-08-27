#FROM nvcr.io/nvidia/deepstream:6.1-triton
FROM nvcr.io/nvidia/deepstream:6.1-devel

RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN pip3 install --upgrade pip
RUN apt update
RUN apt install -y python3-gi python3-dev python3-gst-1.0 python-gi-dev git python-dev \
    python3 python3-pip python3.8-dev cmake g++ build-essential libglib2.0-dev \
    libglib2.0-dev-bin python-gi-dev libtool m4 autoconf automake
WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_python_apps
RUN git submodule update --init && \
apt-get install -y apt-transport-https ca-certificates -y && \
update-ca-certificates && \
cd /opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_python_apps/3rdparty/gst-python/ && ./autogen.sh && make && make install && \
cd /opt/nvidia/deepstream/deepstream-6.1/sources/deepstream_python_apps/bindings && mkdir build && cd build && cmake ..  -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=8 && make && \
apt install -y libgirepository1.0-dev libcairo2-dev && pip3 install ./pyds-*.whl

ARG CLASSES
ARG DISPLAY

ENV DISPLAY=${DISPLAY}

WORKDIR /opt/nvidia/deepstream/deepstream-6.1/sources/dst_python_apps