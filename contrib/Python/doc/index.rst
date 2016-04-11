.. CNTK 1.5 documentation master file, created by
   sphinx-quickstart on Wed Apr  6 13:21:01 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. some aliases
.. _CNTK: http://cntk.ai/

Python for CNTK 1.5
====================================

CNTK_ is a computational toolkit to train and run deep learning networks. It is known for its speed [TODO shootout], because it does not unroll loops (like they occur in RNNs, e.g.) in high-level languages like Python, but instead keeps them in the highly-optimized C++ domain. [TODO improve sales pitch]

To leverage this, one needs to write BrainScript, which might be a bit convoluted (pun intended) at the first encounter. Meet the Python API, which tries to hide away many of the surprises and exposes the graph operators with a NumPy-like interface. Under the hood, the Python API will create the required configuration and input files, run the cntk executable on the graph and then return the result as a NumPy array.

.. note:: CNTK 1.5 is an intermediate step towards CNTK 2.0, which will expose all of its internals to Python. Nevertheless, the Python API should stay stay stable, so that your Python CNTK application should not require many changes once 2.0 has arrived.

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

   CNTK operators <cntk.ops>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. automodule:cntk
