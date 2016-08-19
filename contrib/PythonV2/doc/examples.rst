Examples
========

Logistic Regression
-------------------

Examples for logistic regression you find here: `<https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/LogReg/>`_ 

- Using training and testing data *from a file* : `logreg.py <https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/LogReg/logreg.py>`_ .
- Using training and testing data *from a NumPy array* : `logreg_numpy.py <https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/LogReg/logreg_numpy.py>`_ .

LSTM-based sequence classification
----------------------------------
An Example for training an LSTM-based sequence classification model with embedding you find here: `<https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/LSTM/>`_ .
A typical application would be text classification where we leverage a precomputed word-embedding. 
This is also a good example to see how to provide *input data for sequences* and using *sparse input*.

- In  `Train_sparse.txt <https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/LSTM/>`_  we have two inputs. The input *x* provides the sequence data in sparse form, while *y* provides the classes in dense form.
- The example also uses a predefined embedding (`embeddingmatrix.txt <https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/LSTM/embeddingmatrix.txt>`_ ) mapping each dimension *x* to an embedding vector in a lower dimensional space.

One hidden layer neural network
-------------------------------
 
Example for training a *one hidden layer neural network* using the MNIST-data (recognition of handwritten digits) you find here: `<https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/MNIST/>`_ .

To obtain and prepare the MNIST data use `fetch_mnist_data.py <https://github.com/Microsoft/CNTK/tree/master/contrib/Python/cntk/examples/MNIST/fetch_mnist_data.py>`_ .

