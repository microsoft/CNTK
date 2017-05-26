# MNIST Dataset

The MNIST dataset (http://yann.lecun.com/exdb/mnist/) for handwritten digits recognition is one of the most widely used image dataset for experimenting with different classification algorithms. MNIST has a training set of 60,000 examples, and a test set of 10,000 examples. Each example contains one digit that has been size-normalized and centered in a grayscale image at 28x28 pixel resolution.

The MNIST dataset is not included in the CNTK distribution but can be easily
downloaded and converted to CNTK-supported format by cd to this directory, Examples/Image/DataSets/MNIST  and running the following Python command:

`python install_mnist.py`

After running the script, you will see two output files in the current folder: `Train-28x28_cntk_text.txt` and `Test-28x28_cntk_text.txt`. The total amount of disk space required is around `124`MB. You may now proceed to the [`GettingStarted`](../../GettingStarted) folder to play with this dataset. 

Further, we provide two advanced examples with MNIST. The first one is a [`Multi-Layer Perceptron network (MLP)`](../../Classification/MLP), which achieves about 1.5% error rate. The second one is a [`Convolutional Neural Network (ConvNet)`](../../Classification/ConvNet), which achieves about 0.5% error rate. These results are comparable to the best published results using these types of networks.

If you are curious about how well computers can perform on MNIST today, Rodrigo Benenson maintains a [blog](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354) on the state-of-the-art performance of various algorithms.  
