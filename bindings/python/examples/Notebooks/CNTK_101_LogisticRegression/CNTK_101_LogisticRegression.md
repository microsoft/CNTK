
# CNTK 101: Logistic Regression and ML Primer

This tutorial is targeted to individuals who are new to CNTK and to machine learning. At the end of this tutorial, you would have train a simple yet very powerful machine learning model that is widely used in industry for a variety of applications. The model trained below scales to handle massive data sets in the most expeditious manner by harnessing computational sclability provided by CNTK library leveraging multiple GPUs across multiple machines.

## Introduction

**Problem** (recap from the CNTK 101):
A cancer hospital has provided data and wants us to determine if a patient has a fatal [malignant][] cancer vs. a benign growth. This is what is known as a classification problem. To help classify each patient, we are given their age and the size of the tumor. Intuititely, one can imagine that younger patients and/or patient with small tumor size are less likely to have malignant cancer. The data set simulates this application where the each observation is a patient represented as a dot where red color indicates malignant and blue indicates a benign disease Note: This is a toy example for learning, in real life there are large number of features from different tests/examination sources and doctors'  experience that play into the treatment decision for a patient.

<img src="https://www.cntk.ai/jup/cancer_data_plot.jpg", width=400, height=400>

**Goal**:
Our goal is to learn a classifier that automatically can label any patient into either benign or malignant category given two features (age, tumor size). In this tutorial, we will create a linear classifier that is fundamental to deep networks.

<img src="https://www.cntk.ai/jup/cancer_classify_plot.jpg", width=400, height=400>

In the figure above, the green line is the model learnt from the data and separated the blue dots from the red dots. This tutorial will walk through the steps to learn the green line. Note: this classifer does make mistakes where couple of blue dots on the wrong side of the green line. However, there are ways one can model such that all cancer patients get classified correctly at the cost of classifing some of the begign ones being also incorrectly being labeled as cancer as opposed to the other way around which would leave a lot of cancer patient go un-treated. 

**Approach**: 
Any learning algorithm has typically 5 stages namely, Data Reading, Data Preprocessing, Creating a model, Learning the model parameters and Evaluation (a.k.a. testing/prediction). 

>1. Data reading: We are using a simulated data set that generates two features (plotted below) indicative of the age and tumor size
>2. Data preprocessing: Often the individual features such as size or age needs to be scaled to the same axis. Typically one would scale the data between 0 and 1. To keep things simple, we are not doing any scaling in this tutorial. For more details on [feature scaling][]).
>3. Model Creation: In this tutorial we introduce a basic linear model. Details later in the tutorial.
>4. Learning the model: This is also known as training. While fitting a linear model can be done in a variety of ways including [least squares fit][]. Typically we use Stocastic Gradient Descent a.k.a. sgd (See more on [SGD][])
>5. Evaluation: This is also known as testing where one takes data sets with known labels (a.k.a ground-truth) that was not used ever for training. This allows us to assess, how a model would perform in real world (previously unseen) observations.

## Logistic Regression
Logistic regression is fundamental machine learning technique that uses a linear weighted combination of features and generates the probability of predicting different classes. In our case the classifer will generate a binary label (0 or 1). However, the method shown can be extended to multiple classes. 

<img src="https://www.cntk.ai/jup/logistic_neuron.jpg", width=300, height=200>

In the figure above, contribution from different input features are linearly weighted and aggregated. The resulting sum is mapped to a 0-1 range via a sigmoid function. For classifiers with more than two output labels, one can use a [softmax][] function.

[feature scaling]: https://en.wikipedia.org/wiki/Feature_scaling

[SGD]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent

[least squares fit]: https://en.wikipedia.org/wiki/Least_squares

[softmax]: https://en.wikipedia.org/wiki/Multinomial_logistic_regression



```python
# Import the relevant components
import numpy as np
import sys
import os
from cntk.learner import sgd
from cntk import DeviceDescriptor, Trainer, cntk_device, StreamConfiguration, text_format_minibatch_source
from cntk.ops import input_variable, cross_entropy_with_softmax, combine, classification_error, sigmoid
from cntk.ops import *
```

For this tutorial we use a CPU device. You may run the code on GPU by setting the target target to it. Use: `DeviceDescriptor.gpu_device(0)` instead. 

Known Issue: The `eval` function does not work on GPU device with the input feature map as provided in the tutorial
TODO: Fix eval issue


```python
# Specify the target device to be used for computing (this example is showing for CPU usage)
target_device = DeviceDescriptor.cpu_device()  

if not DeviceDescriptor.default_device() == target_device:
    DeviceDescriptor.set_default_device(target_device) 
```

## Data Generation
Lets generate some syntetic data emulating the cancer example using `numpy` library. We have two features (thus the data is 2 dimensional)  each either being to one of the 2 classes (benign:blue dot or malignant:red dot). 

In our example, each observation in the training data has a label (blue vs. red) corresponding to each observation (set of features - age and size). In this example, we have 2 classes represened by labels 0 or 1, thus a  binary classification task. 


```python
# Define the network
input_dim = 2
num_output_classes = 2
```

### Input and Labels

In this tutorial we are generating synthetic data using `numpy` library. In real world problems, one would put in a reader, that would read a feature matrix (`features`) where each row would represent an obeserved feature set.  Note, the each observation can reside in a higher dimension space and will be represented as a tensor. More advanced tutorials shall introduce the handling of high dimensional data.


```python
#Ensure we always get the same amount of randomness
np.random.seed(0)

#Helper function to generate a random data sample
def generate_random_data(sample_size, feature_dim, num_classes):
    # Create synthetic data using NumPy. 
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

    # Make sure that the data is separable
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)
    X = X.astype(np.float32)    
    # converting class 0 into the vector "1 0 0", 
    # class 1 into vector "0 1 0", ...
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y   
```


```python
# Create the input variables denoting the features and the label data. Note: the input_variable does not need 
# additional info on number of observations (Samples) since CNTK first create only the network tooplogy first 
mysamplesize = 25
features, labels = generate_random_data(mysamplesize, input_dim, num_output_classes)
```

Let's plot the data to see how the input data looks. 
**Caution**: If the import of `matplotlib.pyplot` fails, please run `conda install matplotlib` which will fix the `pyplot` version dependencies


```python
# Plot the data 
import matplotlib.pyplot as plt

#given this is a 2 class 
colors = ['r' if l == 0 else 'b' for l in labels[:,0]]

plt.scatter(features[:,0], features[:,1], c=colors)
plt.show()
```


![png](output_10_0.png)


# Model Creation

A logistic regression (a.k.a LR) network is the simplest building block but has been powering hundereds of ML 
applications in the past decade. LR is a simple linear model that takes an input feature vector ($\bf{x}$) (the blue nodes in the figure) and emits the *evidence* ($\bf{z}$) (output of the green node). Each feature in the input layer is connected with a output node by a corresponding weight w (indicated by the blue lines of varying thickness). 

<img src="https://www.cntk.ai/jup/logistic_neuron.jpg", width=300, height=200>

The first step is to compute the evidence for an observation. 

$$\bf{z} = \bf{w} \times \bf{x} + b$$ 

where $\bf{w}$ is the weight vector of lenght $n$ and $b}$ is a bias. Note: we use **bold** notations to denote vectors notation. 

The computed evidence is mapped to a 0-1 scale using a `sigmoid` or a `softmax` function.

Network input and output: 
- an **input** variable: This is the equal to the dimension of the data, e.g., if a 3D array of depth 10, height 10  and width 5 columns are presented, then the input feature dimension will be 3. [More on data and their dimensions to appear in separate tutorials]. In this case we have two dimensions. Corresponding to each dimension we associate a category.  


```python
input = input_variable((input_dim), np.float32)
label = input_variable((num_output_classes), np.float32)
```

## Network setup

The `linear_layer` function is pretty straight forward and should be easy to implement. We perform two operations:
0. multiply the weights ($\bf{w}$)  with the features ($\bf{x}$),
1. add the bias term $b$.


```python
#QUESTION: input_var.output() construct is wierd; can we hide it
mydict = {"w":[],"b":[]}
def linear_layer(input_var, output_dim):
    try:
        shape = input_var.shape()
    except AttributeError:
        input_var = input_var.output()
        shape = input_var.shape()

    input_dim = shape[0]
    weight_param = parameter(shape=(input_dim, output_dim))
    bias_param = parameter(shape=(output_dim))
    
    mydict['w'] = weight_param
    mydict['b'] = bias_param

    t = times(input_var, weight_param)
    return bias_param + t
```


```python
output_dim = num_output_classes
netout = linear_layer(input, output_dim)
```

### Learning model parameters

Now that the network is setup, we would like to learn the parameters $\bf W$ and $\bf b$ for each of the layers in our network. To do so we convert, the computed evidence ($\bf z_{final~layer}$) into a set of predicted probabilities $\bf y$ using a `softmax` function.

$$ p = softmax(~evidence~)$$ 

One can see the `softmax` function as an activation function that maps the accummulated evidences to a probability distribution over the number of classes, in our example 2 (Details of the [softmax function][]). Other choices of activation function can be [found here][].

[softmax function]: http://lsstce08:8000/cntk.ops.html#cntk.ops.softmax

[found here]: https://github.com/Microsoft/CNTK/wiki/Activation-Functions

## Training
The output of the softmax is a probability of each observation (rows of $\bf x$) belonging to the respective classes. For training the classifer we need to determine what behavior the model needs to mimic. In other words, we want the generated probabilities to be as close as possible to the observed labels. This function is called the *cost* or *loss* fucntion and shows what is the difference between the learnt model vs. that generated by the training set.

[`Cross-entropy`][] is a popular function to measure the loss and has been used actively in information theory. It is defined as:

$$ H(y) = - \sum_j y_j \log (p_j) $$  

where $y$ is our predicted probability from `softmax` function and $y$ represents the ground truth or the label. Understanding the [details][] of this cross-entropy function is highly recommended.

[`cross-entropy`]: http://lsstce08:8000/cntk.ops.html#cntk.ops.cross_entropy_with_softmax
[details]: http://colah.github.io/posts/2015-09-Visual-Information/


```python
loss = cross_entropy_with_softmax(netout, label)
```

#### Evaluation

In order to evaluate the classification, one can compare the output of the network which for each observation emits a vector of evidences (can be converted into probabilities using `softmax` functions) with dimension equal to number of classes.


```python
label_error = classification_error(netout, label)
```

### Configure training

The trainer strives to reduce the `loss` function by different optimization approaches, [Stochastic Gradient Descent][] (`sgd`) being one of the most poplular one. Typically one would start with some random initialization of the model parameters. We would define a sub-set of the training data called the *minibatch* and update the model parameters using `sgd`. As the process is repeated over with different training samples, the model parameters are updated and the error reduces. When the incremental error rates are no longer changing significantly, we claim that our model is trained.

One of the key parameter for optimization is called the *learning_rate*. For now we can think of it as a scaling factor that modulates how much we change the parameters in any iteration. We will be covering more details in later tutorial. 
With this information, we are ready to create our trainer. 

[optimization]: https://en.wikipedia.org/wiki/Category:Convex_optimization
[Stochastic Gradient Descent]: https://en.wikipedia.org/wiki/Stochastic_gradient_descent


```python
# Instantiate the trainer object to drive the model training
learning_rate = 0.02
trainer = Trainer(netout, loss, label_error, [sgd(netout.parameters(), lr=0.02)])
```

First lets create some helper functions that will be needed to visualize different functions associated with training.


```python
from cntk.utils import get_train_eval_criterion, get_train_loss

# Define a utiltiy function to compute moving average sum (
# More efficient implementation is possible with np.cumsum() function
def moving_average(a, w=10) :
    if len(a) < w: 
        return a[:]    #Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss, eval_error = "NA", "NA"

    if mb%frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_error = get_train_eval_criterion(trainer)
        if verbose: print ("Minibatch: {}, Train Loss: {}, Train Error: {}".format(mb, training_loss, eval_error))
        
    return mb, training_loss, eval_error
```

### Run the trainer

We are now ready to train our LR model. We want to decide what data we need to feed into the training engine.

In this example, each iteration of the optimizer will work on 25 samples (25 dots w.r.t. the plot above) a.k.a *minibatch_size*. We would like to train on say 20000 observations. If the number of samples in the data is 10000. Then the trainer will make multiple passes through the data. Note: In real world case, we would be given a certain amount of labeled data (in the context of this example, observation (age, size) and what they mean (benign / malignant)). We would use a large number of observations for training say 70% and set aside the remainder for evaluation of the trained model.

With these parameters we can proceed withe training our simple feed forward network


```python
#Initialize the parameters for the trainer
minibatch_size = 25
num_samples_to_train = 20000
num_minibatches_to_train = num_samples_to_train  / minibatch_size
```


```python
#Run the trainer on and perform model training
training_progress_output_freq = 20

plotdata = {"batchsize":[], "loss":[], "error":[]}

for i in range(0, int(num_minibatches_to_train)):
    features, labels = generate_random_data(minibatch_size, input_dim, num_output_classes)
    # Specify the mapping of input variables in the model to actual minibatch data to be trained with
    trainer.train_minibatch({input : features, label : labels})
    batchsize, loss, error = print_training_progress(trainer, i, training_progress_output_freq, verbose=1)
    
    if not (loss == "NA" or error =="NA"):
        plotdata["batchsize"].append(batchsize)
        plotdata["loss"].append(loss)
        plotdata["error"].append(error)
```

    Minibatch: 0, Train Loss: 0.693147201538086, Train Error: 0.52
    Minibatch: 20, Train Loss: 5.301351318359375, Train Error: 0.6
    Minibatch: 40, Train Loss: 13.784532470703125, Train Error: 0.84
    Minibatch: 60, Train Loss: 4.681844787597656, Train Error: 0.64
    Minibatch: 80, Train Loss: 0.9043153381347656, Train Error: 0.32
    Minibatch: 100, Train Loss: 0.5508702850341797, Train Error: 0.24
    Minibatch: 120, Train Loss: 0.24632339477539061, Train Error: 0.08
    Minibatch: 140, Train Loss: 0.2562468719482422, Train Error: 0.16
    Minibatch: 160, Train Loss: 0.1664549255371094, Train Error: 0.08
    Minibatch: 180, Train Loss: 0.4681434631347656, Train Error: 0.16
    Minibatch: 200, Train Loss: 1.1755970764160155, Train Error: 0.32
    Minibatch: 220, Train Loss: 0.20667490005493164, Train Error: 0.12
    Minibatch: 240, Train Loss: 0.5005390167236328, Train Error: 0.16
    Minibatch: 260, Train Loss: 0.18562889099121094, Train Error: 0.12
    Minibatch: 280, Train Loss: 0.265356502532959, Train Error: 0.08
    Minibatch: 300, Train Loss: 0.1420402145385742, Train Error: 0.04
    Minibatch: 320, Train Loss: 0.1713204002380371, Train Error: 0.04
    Minibatch: 340, Train Loss: 0.17454261779785157, Train Error: 0.08
    Minibatch: 360, Train Loss: 0.5601414489746094, Train Error: 0.16
    Minibatch: 380, Train Loss: 0.19852914810180664, Train Error: 0.08
    Minibatch: 400, Train Loss: 0.2572595405578613, Train Error: 0.12
    Minibatch: 420, Train Loss: 0.05600847244262695, Train Error: 0.0
    Minibatch: 440, Train Loss: 0.5610065078735351, Train Error: 0.2
    Minibatch: 460, Train Loss: 0.24877582550048827, Train Error: 0.12
    Minibatch: 480, Train Loss: 0.046818904876708985, Train Error: 0.04
    Minibatch: 500, Train Loss: 0.3920000076293945, Train Error: 0.12
    Minibatch: 520, Train Loss: 0.027437171936035155, Train Error: 0.0
    Minibatch: 540, Train Loss: 0.0810181999206543, Train Error: 0.04
    Minibatch: 560, Train Loss: 0.43833831787109373, Train Error: 0.08
    Minibatch: 580, Train Loss: 0.3859831619262695, Train Error: 0.08
    Minibatch: 600, Train Loss: 0.45353504180908205, Train Error: 0.2
    Minibatch: 620, Train Loss: 0.41453697204589846, Train Error: 0.16
    Minibatch: 640, Train Loss: 0.08718055725097656, Train Error: 0.04
    Minibatch: 660, Train Loss: 0.11666389465332032, Train Error: 0.04
    Minibatch: 680, Train Loss: 0.2673243522644043, Train Error: 0.16
    Minibatch: 700, Train Loss: 1.0112106323242187, Train Error: 0.28
    Minibatch: 720, Train Loss: 0.4615753555297852, Train Error: 0.16
    Minibatch: 740, Train Loss: 0.055942535400390625, Train Error: 0.0
    Minibatch: 760, Train Loss: 0.3342399597167969, Train Error: 0.12
    Minibatch: 780, Train Loss: 0.4965230560302734, Train Error: 0.08
    


```python
#Compute the moving average loss to smooth out the noise in SGD    

plotdata["avgloss"] = moving_average(plotdata["loss"])
plotdata["avgerror"] = moving_average(plotdata["error"])

#Plot the training loss and the training error
import matplotlib.pyplot as plt

plt.figure(1)
plt.subplot(211)
plt.plot(plotdata["batchsize"], plotdata["avgloss"], 'b--')
plt.xlabel('Minibatch number')
plt.ylabel('Loss')
plt.title('Minibatch run vs. Training loss ')

plt.show()

plt.subplot(212)
plt.plot(plotdata["batchsize"], plotdata["avgerror"], 'r--')
plt.xlabel('Minibatch number')
plt.ylabel('Label Prediction Error')
plt.title('Minibatch run vs. Label Prediction Error ')
plt.show()
```


![png](output_28_0.png)



![png](output_28_1.png)


## Evaluation / Testing 

Now that we have trained the network. Lets evaluate the trained network on data that hasn't been used for training. This is often called \bf{testing}. Lets create some newdata set and evaluate the average error & loss on this set. This is done using `trainer.test_minibatch`. Note the error on this previously unseen data is comparable to training error. This is a **key** check. Should the error be larger than the training error by a large margin, it indicates that the train model will not perform well on data that it has not seen during training. This is known as [overfitting][]. There are several ways to address overfitting that is beyond the scope of this tutorial but CNTK toolkit provide the necessary components to address overfitting.

[overfitting]: https://en.wikipedia.org/wiki/Overfitting



```python
#Generate new data
features, labels = generate_random_data(minibatch_size, input_dim, num_output_classes)

trainer.test_minibatch({input : features, label : labels}) 
```




    0.04



### Checking prediction / evaluation 
For evaluation, we map the output of the network between 0-1 and convert them into probabilities. This suggests what are the chances that each observation is either malignant or benign. We use a softmax function to get the probabilities of each of the class. 


```python
out = softmax(netout)
result =out.eval({input : features})
```

Lets compare the ground truth label with the predictions. The should be in agreement.

**Question:** How many predictions were mislabeled? Can you change the code below it identify which observations were misclassified? 


```python
print("Label    :", np.argmax(labels[:5],axis=1))
print("Predicted:", np.argmax(result[0,:5,:],axis=1))
```

    Label    : [1 1 0 0 0]
    Predicted: [1 1 0 0 0]
    

### Visualization
It is desirable to visualize the results. In this example, the data is conviniently in 2 dimensions and can be plotted. For data with higher dimensions, one projects the data into a lower dimensional plane.


```python
#Model parameters
bias_vector = mydict['b'].value().to_numpy()
weight_matrix = mydict['w'].value().to_numpy()

# Plot the data 
import matplotlib.pyplot as plt

#given this is a 2 class 
colors = ['r' if l == 0 else 'b' for l in labels[:,0]]
plt.scatter(features[:,0], features[:,1], c=colors)
plt.plot([0,bias_vector[0]/weight_matrix[0][1]], [ bias_vector[1]/weight_matrix[0][0], 0], c = 'g', lw=3)
plt.show()
```


![png](output_36_0.png)



```python

```
