# K2vTune: A Workload-aware Configuration Tuning for RocksDB
This repository contains the source code for the paper ["K2vTune: A Workload-aware Configuration Tuning for RocksDB"](https://www.sciencedirect.com/science/article/pii/S0306457323003047) published on [Information Processing & Management](https://www.sciencedirect.com/journal/information-processing-and-management) in January 2024. K2vTune is a workload-aware configuration tuning framework, which can recognize the configuration knobs of RocksDB according to the workload type  and effectively consider multiple performance metrics using our knob2vec method.
## Requirements
- lifelines
- pytorch == 1.7.0
- python >= 3.8
### SMAC library
- https://automl.github.io/SMAC3/main/index.html
#### How to install
<pre>
conda install gxx_linux-64 gcc_linux-64 swig
pip install smac
</pre>

## QuickStart
### Run main.py to train the entire model. Paser explanation as below,
<pre>
target       : target workload number  
tf           : using teacher forcing, if not specify this, the model will be trained by non-teacher forcing  
train        : mode of train  
eval         : mode of train using pre-trained model(.pt)  
model_path   : if using eval mode, add pre-trained model path  
batch_size   : batch size for dataset
hidden_size  : hidden size of the model  
lr           : learning rate of the model
mode         : regression model type ['raw', 'dnn', 'gru', 'attngru']
attn_mode    : attention tyep ['dot', 'general', 'concat', 'bahdanau']
generation   : the counts of generation in Genetic Algorithm  
pool         : size of pool in genetic algorithm
optimization : choose optimization algorithm ['ga', 'smac']
</pre>
* #### Training the model
Please modify arguments in the bash file. See train.sh
```
./main.sh train
```
or
```
python main.py --target ${target_idx} --tf --train --hidden_size ${hidden_size} --lr ${learning_rate} \
--generation ${generation_num} --pool ${pool_num}
```
* #### Training with pre-trained model path
Please modify arguments in the bash file. See eval.sh
```
./main.sh eval
```
or
```
python main.py --target ${target_idx} --tf --eval --model_path ${model_path} \
--generation ${generation_num} --pool ${pool_num}
```
We set the parameters as follows
- `hidden_size = 128`
- `lr = 0.001`
- `generation = 100`
- `pool = 128`
- `mode = 'attngru'`
- `attn_mode = 'general'`
- `optimization = 'ga'`
