mkdir -p logs/
rm -rf logs/*

n=$1
graph=$2
comm=$3
dataset_home=$4

base=/data/home/zekucai/Projects/datasets/
input_graph=$base/$graph/graph.txt
cached_dir=$base/$graph

mpirun -np $n -x GCCL_CONFIG=./configs/gpu${n}.json \
  -x GCCL_COMM_PATTERN=$comm \
  scripts/mpi_wrapper.sh \
  build/gag \
  --feat_size=256 \
  --input_graph=${input_graph} \
  --cached_graph_dir=${cached_dir}
