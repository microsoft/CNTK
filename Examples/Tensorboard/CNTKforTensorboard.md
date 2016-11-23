# Using Tensorboard to display information from CNTK

The `tf_helpers.py` module that resides under `Examples/common` provides the functionality to display a network graph and scalar plots (such as learning rate change over epochs) in Tensorboard (a web applications from TensorFlow). 

## Prerequisites
* Tensorflow. Please, follow the instructions [here](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html). Available options:
	* Pip
	* Virtualenv
	* Anaconda
	* Docker
	* Source

## Functionality
![graph](https://github.com/Microsoft/CNTK/blob/t-alkhar/tensorboard-cntk-connect/Examples/Tensorboard/graph.JPG "Network graph for simple MNIST example") 
![scalar plots](https://github.com/Microsoft/CNTK/blob/t-alkhar/tensorboard-cntk-connect/Examples/Tensorboard/scalar_plots.JPG "Scalar plots generated during the training") 

## Examples
`SequenceToSequence.py`, `SequenceClassification.py` and `SimpleMNIST.py`(with comments) from this folder showcase the abovementioned functionality.
```
cd Examples/Tensorboard
python SimpleMNIST.py
tensorboard --logdir mnist_log --port 6006
```

## Enabling Tensorboard 
1. Import necessary modules
```python
import tensorflow as tf
from Examples.common.tf_helpers import *
```
2.  Create TF session for tensorboard logging
```python
session = tf.Session()
```
3. Using CNTK network create TF graph
```python
netout = fully_connected_classifier_net(...)
create_tensorflow_graph(netout, session.graph)
```
4. Instantiate a TF summary writer
```python
train_writer = tf.train.SummaryWriter(logdir="/path/to/log-directory", graph=session.graph, flush_secs=30)
```
5. In the training loop add scalar summaries to the summary writer
```python
for i in range(0, int(num_minibatches_to_train)):
        mb = reader_train.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)
        
        # Create TF summary messages
        loss_summary = summary_message("training_loss", trainer.previous_minibatch_loss_average)
        eval_summary = summary_message("train_eval_criterion", trainer.previous_minibatch_evaluation_average)
        
        # Add the summaries to the train writer
        train_writer.add_summary(loss_summary, i)
        train_writer.add_summary(eval_summary, i)
```
6. Close the summary writer after the training is done
```python
train_writer.close()
```
7. Launch TensorBoard
```
tensorboard --logdir=path/to/log-directory --port 6006
```