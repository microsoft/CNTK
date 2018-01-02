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
train_file | file path for train, must be set
eval_file | file path for evalute, must be set
field_cnt | field num in dataSet, field index start by 1, must be set
feature_cnt | feature num in dataSet, feat index start by 1, must be set 
init_value | parameter initialization variance
init_method | parameter initialization method, support normal,tnormal,uniform
embed_l2 | l2 regular coefficient for embedding parameter
embed_l1 | l1 regular coefficient for embedding parameter
layer_l2 | l2 regular coefficient for layer parameter
layer_l1 | l1 regular coefficient for layer parameter
embedding_dim | embedding dim, must be set 
layer_sizes | a list, the number of nodes per layer, must be set
layer_activations | a list, activate function per layer, support sigmoid, relu, tanh, must be set
show_step | show train loss per N step, N = show_step
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
