# CNTK Examples: Image/GAN

## Overview

|Data:     |The MNIST dataset (http://yann.lecun.com/exdb/mnist/) of handwritten digits.
|:---------|:---
|Purpose   |This folder contains an implementation of a distributed trainer for the generative model shown in CNTK 206: Part A - Basic GAN with MNIST data(https://cntk.ai/pythondocs/CNTK_206A_Basic_GAN.html).
|Network   |Generative Adversarial Network.
|Training  |Stochastic gradient descent with momentum.
|Comments  |See below.

## Running the example

### Getting the data

Please refer to CNTK 206: Part A - Basic GAN with MNIST data(https://cntk.ai/pythondocs/CNTK_206A_Basic_GAN.html) for the instructions on getting the data and for an introduction to generative models.

### Executing the example

We use MPI to execute the distributed trainer. For this purpose, let's assume that the MINST dataset is available in a location similar to
base_dir\CNTK\Examples\Image\DataSets\MNIST. We can execute the example using the following MPI command.

`mpiexec -n 4 python Basic_GAN_Distributed.py --datadir base_dir\CNTK\Examples\Image\DataSets\MNIST`

Like CNTK 206: Part A, the distributed example also has `fast` mode and `full` mode of execution. The default execution is set to `fast` mode and once completed, it shows the generated images and display the training loss. For the complete execution, please set the variable `isFast = False` in the example.


## Details

A GAN network is composed of two sub-networks, one called the Generator and the other Discriminator, each of which has its own learner and a learning scheduling. The learning schedule for the Discriminator and the Generator need not be the same. For this example, we use two different trainers to train the Discriminator and the Generator of the GAN model. However, should they have the same learning schedule, then we could use a single trainer with multiple learners to train the GAN model. A similar pattern can be used for other networks where two or more different learners perform the parameter learning using different algorithms. This is facilitated by the CNTK Trainer API (https://cntk.ai/pythondocs/cntk.train.trainer.html?highlight=trainer#module-cntk.train.trainer), in which the trainer accepts a list of learners at it constructor.

