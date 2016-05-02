Getting started
===============

Installation
------------
This page will guide you through the following three required steps:

#. Make sure that all Python requirements are met
#. Build and install CNTK
#. Install the Python API and set it up

Requirements
~~~~~~~~~~~~
You will need the following Python packages: 

:Python: 2.7+ or 3.3+
:NumPy: 1.10
:Scipy: 0.17

On Linux a simple ``pip install`` should suffice. On Windows, you will get
everything you need from `Anaconda <https://www.continuum.io/downloads>`_.

Installing CNTK
~~~~~~~~~~~~~~~
Please follow the instructions on `CNTK's GitHub page 
<https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-your-machine>`_. 
After you have built the CNTK binary, find the build location. It will be 
something like ``<cntkpath>/x64/Debug_CpuOnly/cntk``. You will need this for 
the next step.

Installing the Python module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#. Go to ``<cntkpath>/contrib/Python`` and run ``python setup.py install``
#. Set up the environment variable ``CNTK_EXECUTABLE_PATH`` to point to the
   CNTK executable
#. Enjoy Python's ease of use with CNTK's speed::

>>> import cntk as cn
>>> cn.__version__
1.4
>>> with cn.Context('demo', clean_up=False) as ctx:
...     a = cn.constant([[1,2], [3,4]])
...     print(ctx.eval(a + [[10,20], [30, 40]]))
[[11.0, 22.0], [33.0, 44.0]]

In this case, we have set ``clean_up=False`` so that you can now peek into the
folder ``_cntk_demo`` and see what has been created under the hood for you.

Most likely, you will find issues or rough edges. Please help us improve CNTK
by posting any problems to https://github.com/Microsoft/CNTK/issues. Thanks!

Overview and first run
----------------------

CNTK is a powerful toolkit appropriate for everything from complex deep learning 
research to distributed production environment serving of learned models. It is 
also great for learning, however, and we will start with a basic regression example 
to get comfortable with the API. Then, we will look at an area where CNTK shines: 
working with sequences, where we will demonstrate state-of-the-art sequence classification 
with an LSTM (long short term memory network).

First basic use
~~~~~~~~~~~~~~~

Here is a simple example of using the CNTK Python API to learn to separate data into 
two classes using logistic regression::

	import cntk as C
	from cntk.ops import cntk1

	def main():
	    print("test")


blah...


Sequence classification
~~~~~~~~~~~~~~~~~~~~~~~

One of the most exciting areas in deep learning is the powerful idea of recurrent 
neural networks (RNNs). RNNs are in some ways the Hidden Markov Models of the deep 
learning world. They are networks with loops in them and they allow us to model the 
current state given the result of a previous state. In other words, they allow information 
to persist.

A particular type of RNN -- the Long Short Term Memory (LSTM) network -- is exceedingly 
useful and in practice is what we commonly use when implementing an RNN. For more on why 
LSTMs are so powerful, see, e.g. http://colah.github.io/posts/2015-08-Understanding-LSTMs/. 
For our purposes, we will concentrate on the central feature of the LSTM model: the `memory 
cell`. 

.. image:: images/lstm_cell.png
    :width: 400px
    :alt: LSTM cell

The ...

In this example we can think of the LSTM as a layer being added to the network::

	def lstm_layer(output_dim, cell_dim, x, input_dim):    
    
		# use the CNTK operator `past_value` to get the previous state of the LSTM
		prev_state_h = past_value(0, 'lstm_state_h')
		prev_state_c = past_value(0, 'lstm_state_c')
        
		lstm_state_c, lstm_state_h = lstm_func(output_dim, cell_dim, x, input_dim, prev_state_h, prev_state_c)
		lstm_state_c.name = 'lstm_state_c'
		lstm_state_h.name = 'lstm_state_h'

		# return the hidden state
		return lstm_state_h


...


Operators
----------

Readers
----------
