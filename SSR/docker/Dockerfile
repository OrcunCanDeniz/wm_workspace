FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive  
ENV CUDA_HOME=/usr/local/cuda  
ENV PATH=$CUDA_HOME/bin:$PATH  
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH  
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;8.0;8.6+PTX"


RUN apt-get update && apt-get install -y \
      python3.8 python3.8-dev python3-pip python3-setuptools \
      git build-essential libgl1 libglib2.0-0 libsm6 libxext6 \
      nvidia-cuda-toolkit \
      ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1  
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch and OpenMMLab deps
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 \
      -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install \
      mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.1/index.html \
      mmdet==2.14.0 \
      mmsegmentation==0.14.1 \
      timm

WORKDIR /workspace
RUN git clone https://github.com/open-mmlab/mmdetection3d.git \
    && cd mmdetection3d \
    && git checkout -f v0.17.1 \
    && pip3 install -e . \
    && cd ..

RUN pip3 install nuscenes-devkit==1.1.9  
# RUN git clone https://github.com/PeidongLi/SSR.git

CMD ["bash"]
