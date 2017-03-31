# Build Neural Language Model using Sampled Softmax

This example demonstrates how to use sampled softmax for training a token based neural language model.
The model predicts the next word in a text given the previous ones where the probability of the next word is computed using a softmax.
As the number of different words might be very high this final softmax step can turn out to be costly.

Sampled-softmax is a technique to reduce this cost at training time. For details see also the [sampled softmax tutorial](https://github.com/Microsoft/CNTK/blob/v2.0.rc1/Tutorials/CNTK_207_Training_with_Sampled_Softmax.ipynb)

Note the provided data set has only 10.000 distinct words. This number is still not very high and sampled softmax doesn't show any significant perf improvements here.
The real perf gains will show up with larger vocabularies.

## HOWTO

This example uses Penn Treebank Data which is not stored in GitHub but must be downloaded first.
To download the data please run download_data.py once. This will create a directory ./ptb that contains all the data we need 
for running the example.

Run word-rnn.py to train a model.
The main section of word-rnn defines some parameters to control the training.

* `use_sampled_softmax` allows to switch between sampled-softmax and full softmax.
* `softmax_sample_size` sets the number of random samples used in sampled-softmax. 
