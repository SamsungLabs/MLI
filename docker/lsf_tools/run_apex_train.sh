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
    bash -c "export PYTHONPATH=/home/Develop/MLI; export HOME=/home/Develop/MLI; pip install numpy==1.20.2;
    cd /home/Develop/MLI; FAIL_WITH_NAN=${FAIL_WITH_NAN} python -m torch.distributed.launch --nproc_per_node=${3:-1} --master_port=1234 bin/train.py --config ${1} --output-path ${2:-.}"
