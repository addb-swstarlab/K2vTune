#!/bin/bash

MODE=$1

if [[ $MODE = "train" ]];
then
    # for i in {22..28}
    for i in {16..21}
    do
        sh train.sh $i attngru general
    done
    

elif [[ $MODE = "eval" ]];
then
    # python main.py --target 16 --tf --mode attngru --eval --model_path 20220723/model-20220723-148.pt --hidden_size 128 --generation 100 --optimization smac
    # sh eval.sh 16 attngru 20220723/model-20220723-148.pt smac
    # sh eval.sh 17 attngru 20220724/model-20220724-84.pt smac
    # sh eval.sh 18 attngru 20220723/model-20220723-196.pt smac
    # sh eval.sh 19 attngru 20220723/model-20220723-58.pt smac
    # sh eval.sh 20 attngru 20220723/model-20220723-65.pt smac
    # sh eval.sh 21 attngru 20220723/model-20220723-142.pt smac

    sh eval.sh 16 attngru 20220723/model-20220723-148.pt ga
    # sh eval.sh 17 attngru 20220724/model-20220724-84.pt ga
    # sh eval.sh 18 attngru 20220723/model-20220723-196.pt ga
    sh eval.sh 19 attngru 20220723/model-20220723-58.pt ga
    # sh eval.sh 20 attngru 20220723/model-20220723-65.pt ga
    # sh eval.sh 21 attngru 20220723/model-20220723-142.pt ga

    # sh eval.sh 16 raw 20220723/model-20220723-36.pt
    # sh eval.sh 16 dnn 20220724/model-20220724-39.pt
    # sh eval.sh 16 gru 20220723/model-20220723-02.pt
    # sh eval_bi.sh 16 gru 20220723/model-20220723-75.pt
    # sh eval.sh 16 attngru 20220723/model-20220723-148.pt
    # sh eval_bi.sh 16 attngru 20220723/model-20220723-185.pt

    # sh eval.sh 17 raw 20220724/model-20220724-44.pt
    # sh eval.sh 17 dnn 20220723/model-20220723-79.pt
    # sh eval.sh 17 gru 20220723/model-20220723-08.pt
    # sh eval_bi.sh 17 gru 20220723/model-20220723-189.pt
    # sh eval.sh 17 attngru 20220724/model-20220724-84.pt
    # sh eval_bi.sh 17 attngru 20220723/model-20220723-155.pt

    # sh eval.sh 18 raw 20220723/model-20220723-84.pt
    # sh eval.sh 18 dnn 20220723/model-20220723-229.pt
    # sh eval.sh 18 gru 20220723/model-20220723-50.pt
    # sh eval_bi.sh 18 gru 20220724/model-20220724-89.pt
    # sh eval.sh 18 attngru 20220723/model-20220723-196.pt
    # sh eval_bi.sh 18 attngru 20220724/model-20220724-55.pt

    # sh eval.sh 19 raw 20220723/model-20220723-198.pt
    # sh eval.sh 19 dnn 20220723/model-20220723-163.pt
    # sh eval.sh 19 gru 20220724/model-20220724-58.pt
    # sh eval_bi.sh 19 gru 20220723/model-20220723-57.pt
    # sh eval.sh 19 attngru 20220723/model-20220723-58.pt
    # sh eval_bi.sh 19 attngru 20220723/model-20220723-23.pt

    # sh eval.sh 20 raw 20220724/model-20220724-62.pt
    # sh eval.sh 20 dnn 20220723/model-20220723-241.pt
    # sh eval.sh 20 gru 20220723/model-20220723-62.pt
    # sh eval_bi.sh 20 gru 20220723/model-20220723-207.pt
    # sh eval.sh 20 attngru 20220723/model-20220723-124.pt
    # sh eval_bi.sh 20 attngru 20220724/model-20220724-103.pt

    # sh eval.sh 21 raw 20220723/model-20220723-102.pt
    # sh eval.sh 21 dnn 20220723/model-20220723-247.pt
    # sh eval.sh 21 gru 20220723/model-20220723-212.pt
    # sh eval_bi.sh 21 gru 20220723/model-20220723-213.pt
    # sh eval.sh 21 attngru 20220723/model-20220723-142.pt
    # sh eval_bi.sh 21 attngru 20220723/model-20220723-107.pt
fi