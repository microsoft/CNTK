Concepts 
========

There is a common property in key machine learning models, such as deep neural
networks (DNNs), convolutional neural networks (CNNs), and recurrent neural 
networks (RNNs). All of these models can be described as *computational networks*.

The directed edges of these *computational networks* are vectors, matrices, or in 
general n-dimensional arrays (tensors) which represent input data and model 
parameters. The vertices are *functions* (also called operations) that are 
performing a computation on these input tensors. 


Tensors
-------

The underlying data structure in CNTK is that of a *tensor*. It is a 
multidimensional array on which computations can be performed. Every dimension in 
these arrays is referred to as an *axis* to distinguish it from the scalar size 
of every axis. So, a matrix has two *axes* which both have a certain 
*dimension* corresponding to the number of rows and columns of the *axes*. 

Using tensors makes the framework generic in that it can be used e.g. for 
classification problems where the inputs are vectors, black-and-white 
images (input is a matrix of points), color images (includes a separate dimension 
for r, g, and b) or videos (has an extra time dimension). 

- Tensors have a *shape* which describes the dimensions of its axes. E.g. a shape ``[2,3,4]`` 
  would refer to a tensor with three axes that have, respectively, 2, 3, and 4 
  dimensions. 

- CNTK allows for the last axis to be a *dynamic axis*, i.e. an axis whose size 
  might vary between input samples. This allows for easily 
  modelling sequences (for recurrent networks) without needing to introduce masks 
  or padding. See below for a detailed explanation.

- All data inside of a tensor is of a certain data type. Right now, CNTK 
  implements *float* (32 bit) and *double* (64 bit) precision floating point types, 
  and all tensors in a network have the same type.

- Tensors come either in *dense* or *sparse* form. Sparse tensors should be used
  whenever the bulk of its values are 0. The Python API supports sparse
  tensors, however, the data ingestion of sparse tensors is only supported via
  the reader framework and not yet through NumPy.

  
Tensors are introduced in CNTK in one of three places:

- **Inputs**: These represent data inputs to the computation which are usually 
  bound to a data reader. Data inputs are organized as (mini) batches and 
  therefore receive an extra minibatch dimension. In addition, inputs can have a 
  "ragged" axis called "dynamic axis" which is used to model sequential data. See 
  below for details.

- **Parameters**: Parameters are weight tensors that make up the bulk of the 
  actual model. Parameters are initialized using a constant (e.g. all 0's, 
  randomly  generated data, or initialized from a file) and are updated during 
  *backpropagation* in a training run.

- **Constants**: Constants are very similar to parameters, but they are not 
  taking part in backpropagation.

All of these represent the *leaf nodes* in the network, or, in other words, the 
input parameters of the function that the network represents.

To introduce a tensor, simply use one of the methods in the cntk namespace. Once 
introduced, overloaded operators can be applied to them to form an operator graph::

  import cntk as C

  # Create an input with the shape (2,3,*)
  >>> x = C.input_variable((2,3), name='features') 

  # Create a constant scalar with value 2
  >>> c = C.constant(value=2)

  # Create a parameter of shape (2,3), randomly initialized
  >>> w = C.parameter((2,3))         

  # Set up some test input data to check the operators.
  # We specify a full batch having a sequence with one element, which is a
  # (2,3) matrix.
  >>> test_input = [[ np.asarray([[10,20,30],[40,50,60]]) ]]

  # Elementwise multiplication operation
  >>> op  = x * c                    

  # Evaluate the op using test_input
  >>> print(op.eval({ x: test_input }))
  [[[[  20.   40.   60.]
     [  80.  100.  120.]]]]
     
  # Same as above (2 will be converted to constant)
  >>> op2 = x * 2                    
  >>> print(op2.eval({ x: test_input }))
  [[[[  20.   40.   60.]
     [  80.  100.  120.]]]]

  #  Elementwise multiplication of two 2x3 matrices 
  >>> op3 = x * [[1,2,3], [4,5,6]]  
  >>> print(op3.eval({ x: test_input}))
  [[[[  10.   40.   90.]
     [ 160.  250.  360.]]]]

Broadcasting
~~~~~~~~~~~~

For operations that require the tensor dimensions of their arguments to match, 
*broadcasting*  is applied automatically whenever a tensor dimension is 1. 
Examples are elementwise product or plus operations.
E.g. the following are equivalent:

  >>> C.element_times([2,3], 2).eval()
  array([ 4.,  6.], dtype=float32)

  >>> C.element_times([2,3], [2,2]).eval()
  array([ 4.,  6.], dtype=float32)
  
  
