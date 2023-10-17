#!/bin/bash

if [ $# = 4 ] ; then
    python main.py --target $1 --tf --mode $2 --eval --model_path $3 --hidden_size 128 --generation 100 --optimization $4
fi
