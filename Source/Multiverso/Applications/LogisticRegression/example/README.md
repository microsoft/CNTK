This is a simple example for running [MNIST](http://yann.lecun.com/exdb/mnist/) data set with a multiple classification task using no parameter server.

In Linux just run `sh run.sh`. This script will do build project, download data, convert data format and run the project.

In windows, you can build the project and download the data set, then run `python convert.py` to convert the data format and use the `mnist.config` as the command argument to start an program instance.
