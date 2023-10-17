#!/bin/bash

MODE=$1

if [[ $MODE = "train" ]];
then
    for i in {16..21}
    do
        sh train.sh $i attngru general
    done
    

elif [[ $MODE = "eval" ]];
then
    sh eval.sh 16 attngru 20220723/model-20220723-148.pt ga
fi
