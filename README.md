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


SIMPLI model already in the repo. 

## Train model with sword dataset 
dataset link - https://samsunglabs.github.io/StereoLayers/sword/ 
You need to load the full dataset as the small test data. 
Then set the correct paths to data in the config file. 

```
torchrun \
  --standalone \ # if all GPUs are on a single node
  --nproc_per_node=1 \ # set number of GPUs here
  bin/train.py \
  --config configs/tblock4_train.yaml \
  --output-path train_outputs
```  

## Prepare validation data

`render_val_dataset.py` script below expects a different data format (check out `lib/datasets/val_dataset.py` for details). First, process your data with configs from `configs/val_data`: `sword_val.yaml` is for rendering the exact copies of the validation dataset, `sword_val_spiral.yaml` is for generating views along novel trajectories. Possible trajectories are listed in `lib/modules/cameras/trajectory_generators.py`.
The following script prepares data for validation format.

```
python bin/val_utils/generate_val_dataset.py \
        --config configs/val_data/sword_val_spiral.yaml \
        --output-path sample_val_spiral
```

## Render predefined Dataset

Weights for rendering the model must be in `checkpoints` subfolder of the same folder as the config.

```
python bin/val_utils/render_val_dataset.py \
       --config pretrained/stereo_layers/stereo_layers.yaml \
       --val-dataset datasets/sword_sample/ \ #folder of the data in validation format
       --iteration 400000 \ #iteration of the checkpoint, must be in its file name
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

