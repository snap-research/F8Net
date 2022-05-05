#!/bin/bash
cfg=$1
bs=${2:-256}
num_gpus=${3:-8}
num_proc_per_nodes=$(( num_gpus < 8 ? num_gpus : 8 ))
echo "Total batch size: " $bs
echo "No. of processes per node: " $num_proc_per_nodes
if [ ! -f $cfg ]; then
    echo "Config not found!"
fi

RANK=0 python3 -W ignore -m torch.distributed.launch --nproc_per_node=$num_proc_per_nodes fix_train.py app:$cfg bs:$bs
