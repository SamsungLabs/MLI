# SIMPLI -  Self-improving Multiplane-to-layer Images for Novel View Synthesis
------------
  <p align="center">
  <br>
<a href='https://colab.research.google.com/drive/1EwLRJIzUdGPNNrIxp-JX_4uX6dKxORnI?usp=sharing' style='padding-left: 0.5rem;'><img src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a>
  </br>
 <h2 align="center">WACV 2023</h2>
  <div align="center">
    <img src="./docs/static/images/cover.gif" alt="Logo" width="100%">
  </div>

We suggest to use collab to run our inference code.
To see the resulted geometry use viewer incorporated into the notebook.
We demonstrate results at our [project page](https://samsunglabs.github.io/MLI/).

## Build docker image
After downloading the repository, use `docker/Dockerfile` to create an image that sets up all dependencies.
```sh
docker build -t nvs_torch_image ./docker

nvidia-docker container run \
    -it --shm-size=16G \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    --rm \
    -v $(pwd):/home/Develop/nvs_torch  \
    nvs_torch_image:latest
```

## Setup env
```sh
cd /home/Develop/nvs_torch
export PYTHONPATH=/home/Develop/nvs_torch/
export TORCH_EXTENSIONS_DIR=tmp
export MPLCONFIGDIR=tmp
```


## Download pretrained StereoLayers model
```sh
./pretrained/download.sh
```

SIMPLI model already in the repo. 

## Train model with sword dataset 
```
torchrun \
  --standalone \ # if all GPUs are on a single node
  --nproc_per_node=1 \ # set number of GPUs here
  bin/train.py \
  --config configs/tblock4_train.yaml \
  --output-path train_outputs
```  

## Render predefined Dataset
```
python bin/val_utils/render_val_dataset.py \
       --config pretrained/stereo_layers/stereo_layers.yaml \
       --val-dataset datasets/sword_sample/ \
       --iteration 400000 \
       --output-path outputs
```

## Run Rendering with custom functions
```
python bin/render.py \
       --config pretrained/stereo_layers/stereo_layers.yaml \
       --checkpoints-path pretrained/stereo_layers/checkpoints \
       --iteration 400000 \
       --output-path outputs
```

