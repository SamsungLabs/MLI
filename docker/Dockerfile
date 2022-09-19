FROM nvidia/cudagl:11.3.0-devel-ubuntu18.04

ARG USERNAME=docker

ENV DEBIAN_FRONTEND noninteractive
## Base packages for ubuntu

# clean the libs list
RUN apt-get clean
RUN apt-get update -qq
RUN apt-get install -y \
    pkg-config \
    git \
    wget \
    bzip2 \
    htop \
    vim \
    nano \
    g++ \
    make \
    build-essential \
    software-properties-common \
    apt-transport-https \
    sudo \
    gosu \
    libgl1-mesa-glx \
    graphviz \
    curl \
    libglew-dev \
    libglfw3-dev \
    libjpeg-dev \
    libjsoncpp-dev \
    libpng-dev \
    lxde \
    mesa-utils \
    x11vnc \
    xorg-dev \
    xvfb \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    ne \
    x264


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl


## Download and install miniconda

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh -O /tmp/miniconda.sh

RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

## Install base conda packages

RUN conda install -y numpy scipy pandas matplotlib pillow scikit-learn

RUN conda list

## Torch
RUN conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath

## OpenCV

RUN conda install -c conda-forge opencv
RUN conda install x264 ffmpeg -c conda-forge

## Additional packages

RUN pip install tqdm \
                h5py \
                scikit-image \
                einops \
                graphviz \
                plyfile \
                munch \
                huepy \
                Cython \
                tensorboardX \
                packaging \
                torchsummary \
                imageio \
                pyyaml \
                opencv-python \
                pyntcloud \
                lpips \
                kornia==0.5.11


#RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu102_pyt171/download.html
RUN conda install pytorch3d -c pytorch3d
#
ENV FORCE_CUDA="1"
ENV CUDA_VISIBLE_DEVICES=0
# RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git"


# we don't install any libs for prepare_sequence scripts here

RUN pip install \
                tensorflow \
                tensorboardX \
                pytorch-memlab \
                albumentations \
                numpy==1.20.2

#RUN pip install \
#                PyOpenGL \
#                PyOpenGL_accelerate \
#                trimesh



# Add non-root user
ARG USERNAME=docker
RUN apt-get update && apt-get install -y sudo curl && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\n" > /etc/fixuid/config.yml

# Install ceres solver
WORKDIR /opt
RUN git clone https://github.com/NVlabs/nvdiffrast.git && \
    cd nvdiffrast && \
    pip install .

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN pip install ninja

RUN mkdir -p /home/docker/.cache/torch/hub/checkpoints/
WORKDIR /home/docker/.cache/torch/hub/checkpoints/
RUN wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
RUN wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
RUN chmod 777 /home/docker/.cache/torch/hub/checkpoints/

WORKDIR "/home/$USERNAME"
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
USER $USERNAME:$USERNAME

ENTRYPOINT ["fixuid", "-q"]
CMD ["fixuid", "-q", "bash"]
