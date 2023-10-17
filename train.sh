# mode = raw, dnn, gru, attngru
# attn_mode = dot, general, concat, bahdanau

## For training new version

if [ $# = 4 ] ; then
    python main.py --target $1 --mode $2 --attn_mode $3 --tf --train --hidden_size 128 --lr 0.001 --generation 100 --save --optimization $4
fi
