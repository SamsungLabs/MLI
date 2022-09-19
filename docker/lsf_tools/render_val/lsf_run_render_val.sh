config_path="${1}"
val_dataset="${2}"
iteration="${3}"
scale="${4}"
config_file="${config_path##*/}"
config_name="${config_file%.yaml*}"
username="$(whoami)"
mkdir -p job_logs
bsub \
    -o job_logs/render.%J.out \
    -e job_logs/render.%J.err \
    -J render \
    -q "${6}" \
    -gpu "num=1:mode=exclusive_process" \
    -m "${5}" \
    "./docker/lsf_tools/render_val/run_render_val.sh ${config_path} ${val_dataset} ${iteration} ${scale} /Vol1/dbstore/datasets/multimodal/PsinaOutputs/CVPR2022_validation"

