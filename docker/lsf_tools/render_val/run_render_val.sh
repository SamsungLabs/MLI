export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NV_GPU=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr '\n' ',')
LSB_CONTAINER_IMAGE=t.khakh/psina3d NV_GPU=${NV_GPU} nvidia-docker container run \
    -i --shm-size=16G \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    --rm \
    -v /Vol1/:/Vol1/ \
    -v /Vol0/:/Vol0/ \
    -v ~/Develop/MLI/:/home/Develop/MLI/  \
    airuhead01:5000/t.khakh/psina3d:latest \
    bash -c "export PYTHONPATH=/home/Develop/MLI; export HOME=/home/Develop/MLI
    cd /home/Develop/MLI; pip install numpy==1.20.2;
    python bin/val_utils/render_val_dataset.py --config ${1} --val-dataset ${2} --iteration ${3} --scene-scale-constant ${4} --output-path ${5}"
