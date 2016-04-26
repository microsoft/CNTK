
.. some aliases
.. _CNTK: http://cntk.ai/

Python for CNTK 1.4
===================

CNTK_ is a computational toolkit to train and run deep learning networks. It 
is known for its `speed 
<https://blogs.technet.microsoft.com/inside_microsoft_research/2015/12/07/microsoft-computational-network-toolkit-offers-most-efficient-distributed-deep-learning-computational-performance/>`_, 
because it does not unroll loops (like they occur in RNNs, e.g.) in high - level
languages like Python, but instead keeps them in the highly - optimized 
C + + domain.

Meet the Python API, which tries to give you a smooth ride towards fast deep 
learning training and exposes the graph operators with a NumPy-like interface.  
Under the hood, the Python API will create the required configuration and 
input files, run the cntk executable on the graph and then return the result 
as a NumPy array. 

.. note:: CNTK 1.4 is an intermediate step towards CNTK 2.0, which will expose all of its internals to Python. Nevertheless, the Python API should stay stable, so that your Python CNTK application should not require many changes once CNTK 2.0 has arrived.

Example
#######

.. code-block:: python
    :linenos:

    import cntk as cn

    with cn.Context('demo') as ctx:
        a = cn.constant([[1,2], [3,4]])
        out = a + [[10,20], [30, 40]]

        result = ctx.eval(out)

        return result


Contents:
    
.. toctree::
   :maxdepth: 2

   Installation <installation>
   Getting Started <gettingstarted>
   Readers <readers>
   Operators <cntk.ops>
   Examples <examples>





Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. automodule:cntk
