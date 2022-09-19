export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',')
LSB_CONTAINER_IMAGE=t.khakh/psina3d NV_GPU=${NV_GPU} nvidia-docker container run \
    -i --shm-size=16G \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    --rm \
    -v /Vol1/:/Vol1/ \
    -v /Vol0/:/Vol0/ \
    -v ~/Develop/nvs-torch/:/home/Develop/nvs-torch/  \
    airuhead01:5000/t.khakh/psina3d:latest \
    bash -c "export PYTHONPATH=/home/Develop/nvs-torch; export HOME=/home/Develop/nvs-torch
    cd /home/Develop/nvs-torch; pip install numpy==1.20.2;
    python bin/val_utils/render_val_dataset.py --config ${1} --val_dataset ${2} --iteration ${3} --scene_scale_constant ${4} --output_path ${5}"
