n_gpu=${4:-1}
config_path="${1}"
config_file="${config_path##*/}"
config_name="${config_file%.yaml*}_${n_gpu}gpu"
username="$(whoami)"
mkdir -p job_logs
bsub \
    -o "job_logs/${config_name}.%J.out" \
    -e "job_logs/${config_name}.%J.err" \
    -J ${config_name} \
    -q "${3}" \
    -gpu "num=${4:-1}:mode=exclusive_process" \
    -m "${2}" \
    "FAIL_WITH_NAN=${FAIL_WITH_NAN} ./docker/lsf_tools/run_apex_train.sh ${config_path} /Vol1/dbstore/datasets/multimodal/PsinaOutputs/${username} ${n_gpu}"

