#!/bin/bash
cfg=$1
bs=${2:-256}
echo "Total batch size: " $bs
if [ ! -f $cfg ]; then
    echo "Config not found!"
fi

python3 -W ignore fix_train.py app:$cfg bs:$bs
