Execution Context
=================

An execution context defines how and where a CNTK network will be created and run. 
The context can be either `Local` or `Deferred`. In the former case the functions 
(such as training and testing the network) are done locally and immediately so that 
the result is returned interactively to your Python session.

With a `Deferred` context, the functions simply set up a configuration file that can 
be used with CNTK at a later date. For example, if you would like to develop your 
network locally to get things working, and then launch the training on a GPU cluster, 
you can use the deferred context to simply turn your Python script into a CNTK configuration 
and then send that configuration to your cluster.


Usage
-------

.. autoclass:: cntk.context.LocalExecutionContext
   :members:

.. autoclass:: cntk.context.DeferredExecutionContext
   :members:
