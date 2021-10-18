# Knob-Representation
We propose the methods to convert scalar knobs to vector representation

## TRAIN
### Run main.py to train the entire model. Paser explanation as below,
<pre>
target      : target workload number  
tf          : using teacher forcing, if not specify this, the model will be trained by non-teacher forcing  
train       : mode of train  
eval        : mode of train using pre-trained model(.pt)  
model_path  : if using eval mode, add pre-trained model path  
hidden_size : hidden size of the model  
lr          : learning rate of the model  
ex_weight   : balance weight for computing external metrics score and its summation must be 1  
generation  : the counts of generation in Genetic Algorithm  
</pre>
* #### Training the model
```
python main.py --target {target number} --tf --train --hidden_size {hidden size} --lr {learning rate} \
--generation {generation number in genetic algorithm} --ex_weight 0.25 --ex_weight 0.25 --ex_weight 0.25 --ex_weight 0.25 
```
* #### Training with pre-trained model path
```
python main.py --target {target number} --tf --eval --model_path {model_path} \
--generation {generation number in genetic algorithm} --ex_weight 0.25 --ex_weight 0.25 --ex_weight 0.25 --ex_weight 0.25 
```

## Updated History
* (211018) update data/README.md to explain data samples and README.md how to train the model
