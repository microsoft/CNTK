## What is Wide & Deep Model 
Wide & Deep Model jointly trained wide linear models and deep neural networks—to combine the benefits of memorization and generalization for recommender systems
## Data Format
we support data format [libffm](https://github.com/guestwalk/libffm)

## How To Use

firstly, in order to train model on your own data, you can set your own parameters at params in deepWide.py.
```
    params = {
        'feature_cnt': 194081,
        'field_cnt': 33,
        'batch_size': 3,
        'train_file': './data/train.entertainment.no_inter.norm.fieldwise.userid.txt',
        'eval_file': './data/val.entertainment.no_inter.norm.fieldwise.userid.txt',
        'embedding_dim': 10,
        'layer_l2': 0.1,
        'embed_l2': 0.1,
        'init_value': 0.001,
        'init_method': 'normal',
        'n_epochs': 5,
        'learning_rate': 0.01,
        'show_steps': 20,
        'layer_sizes': [50],
        'layer_activations': ['relu']
    }

``` 
then
```
python deepWide.py
``` 

#### Data Parameter
parameter |description | 
----|------| 
train_file | file path for train, must be set, eg, train_file = ./data/train.entertainment.no_inter.norm.fieldwise.userid.txt
eval_file | file path for evalute, must be set, eg, eval_file = ./data/val.entertainment.no_inter.norm.fieldwise.userid.txt
field_cnt | field num in dataSet, field index start by 1, must be set, eg, field_cnt=33
feature_cnt | feature num in dataSet, feat index start by 1, must be set , eg, feature_cnt=194081
init_value | parameter initialization variance, eg, init_value=0.001
init_method | parameter initialization method, support normal,tnormal,uniform, eg, init_method=normal
embed_l2 | l2 regular coefficient for embedding parameter, eg, embed_l2=0.1
layer_l2 | l2 regular coefficient for layer parameter, eg, layer_l2=0.1
embedding_dim | embedding dim, must be set, eg, embedding_dim=10
layer_sizes | a list, the number of nodes per layer, must be set, eg, layer_sizes=[50]
layer_activations | a list, activate function per layer, support sigmoid, relu, tanh, must be set, eg, layer_activations=['relu']
show_steps | show train loss per N step, N = show_step, eg, show_steps=20

## Output And Result

during training, the value of loss function will be output. such as:
``` 
minibatch: 20, loss: 0.6557, logloss: 0.6557 
minibatch: 40, loss: 0.2838, logloss: 0.2835 
minibatch: 60, loss: 0.6476, logloss: 0.6470 
epoch: 0, train logloss: 0.3875, eval logloss: 0.3919 
minibatch: 80, loss: 0.1607, logloss: 0.1601 
minibatch: 100, loss: 0.1958, logloss: 0.1952 
minibatch: 120, loss: 0.1524, logloss: 0.1516 
epoch: 1, train logloss: 0.3691, eval logloss: 0.3880 
``` 
log information is saved in log.txt. such as:
``` 
epoch: 0, train logloss: 0.3875, eval logloss: 0.3919
epoch: 1, train logloss: 0.3691, eval logloss: 0.3880
epoch: 2, train logloss: 0.3473, eval logloss: 0.3876
epoch: 3, train logloss: 0.3255, eval logloss: 0.3874
epoch: 4, train logloss: 0.3058, eval logloss: 0.3872
``` 


## Benchmark Experiment
we sample 8w from criteo dataset([dataset](https://www.kaggle.com/c/criteo-display-ad-challenge)), dealing with long tail features and continuous features. the dataset has 26w features and 8w samples.we split the dataset randomly into two parts: 80% is for training, 20% is for testing.

model |logloss | 
----|------| 
deepFM(tensorflow) | 0.4976 | 
deepWide(CNTK) | 0.4980 | 
fm(tensorflow) | 0.4994 |  
lr(tensorflow) | 0.5008 |   

## References
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
