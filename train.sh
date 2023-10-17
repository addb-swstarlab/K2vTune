# mode = raw, dnn, gru, attngru
# attn_mode = dot, general, concat, bahdanau

# python main.py --target 18 --tf --eval --model_path 20211007/model-20211007-30.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 17 --tf --train --generation 100

# python main.py --target 17 --mode dnn --tf --train --hidden_size 128 --lr 0.0001 --generation 500 \
#     --ex_weight 0.2 --ex_weight 0.4 --ex_weight 0.2 --ex_weight 0.2

# python main.py --target $1 --mode attngru --attn_mode $2 --tf --eval --model_path 20210928/model-20210928-61.pt --hidden_size 128 --lr 0.001 --generation 1000 \
#     --ex_weight 0.25 --ex_weight 0.25 --ex_weight 0.25 --ex_weight 0.25
# 20211006/model-20211006-111.pt

# python main.py --target 16 --tf --mode raw --eval --model_path 20211006/model-20211006-111.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 16 --tf --mode dnn --eval --model_path 20211006/model-20211006-112.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 16 --tf --mode gru --eval --model_path 20211006/model-20211006-113.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 16 --tf --mode attngru --eval --model_path 20211006/model-20211006-114.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 16 --tf --mode attngru --eval --model_path 20211007/model-20211007-09.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 16 --tf --mode attngru --eval --model_path 20211007/model-20211007-08.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 16 --tf --mode attngru --eval --model_path 20211007/model-20211007-14.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1


# python main.py --target 17 --tf --mode raw --eval --model_path 20211006/model-20211006-118.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 17 --tf --mode dnn --eval --model_path 20211007/model-20211007-15.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 17 --tf --mode gru --eval --model_path 20211007/model-20211007-07.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 17 --tf --mode attngru --eval --model_path 20211006/model-20211006-121.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 17 --tf --mode attngru --eval --model_path 20211007/model-20211007-21.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 17 --tf --mode attngru --eval --model_path 20211006/model-20211006-123.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 17 --tf --mode attngru --eval --model_path 20211007/model-20211007-24.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1


# python main.py --target 18 --tf --mode raw --eval --model_path 20211006/model-20211006-125.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 18 --tf --mode dnn --eval --model_path 20211006/model-20211006-126.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 18 --tf --mode gru --eval --model_path 20211006/model-20211006-127.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 18 --tf --mode attngru --eval --model_path 20211007/model-20211007-27.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 18 --tf --mode attngru --eval --model_path 20211006/model-20211006-129.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 18 --tf --mode attngru --eval --model_path 20211007/model-20211007-36.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1

# python main.py --target 18 --tf --mode attngru --eval --model_path 20211007/model-20211007-30.pt --generation 500 --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1


### For training
# python main.py --target $1 --mode $2 --attn_mode $3 --tf --train --hidden_size 128 --lr 0.001 --generation 100 \
    # --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1 --step --sample_size $4 --batch_size $5 --GA_batch_size $5
# python main.py --target $1 --mode $2 --attn_mode $3 --tf --train --hidden_size 128 --lr 0.001 --generation 100 \
#     --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1 --save

## For training new version

if [ $# = 4 ] ; then
    python main.py --target $1 --mode $2 --attn_mode $3 --tf --train --hidden_size 128 --lr 0.001 --generation 100 --save --optimization $4
fi

# if [ $# = 3 ] ; then
#     python main.py --target $1 --mode $2 --attn_mode $3 --tf --train --hidden_size 128 --lr 0.001 --generation 100 \
#     --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1 --save
# fi
# if [ $# = 2 ] ; then
# 	python main.py --target $1 --mode $2 --tf --train --hidden_size 128 --lr 0.001 --generation 100 \
#     --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1 --save
# fi

### For eval mode
# python main.py --target $1 --tf --mode $2 --eval --model_path $3 --hidden_size 128 --generation 100 \
#     --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1 #--step

# if [ $# = 3 ] ; then
#     python main.py --target $1 --tf --mode $2 --eval --model_path $3 --hidden_size 128 --generation 100 \
#     --ex_weight 1 --ex_weight 1 --ex_weight 1 --ex_weight 1 --step
# fi

