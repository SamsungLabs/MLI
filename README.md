# Self-improving Multiplane-to-layer Images for Novel View Synthesis
------------


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

