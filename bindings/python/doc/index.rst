
.. some aliases
.. _CNTK: https://cntk.ai/

Python API for CNTK (2.0rc1)
==================================

CNTK_, the Microsoft Cognitive Toolkit, is a system for describing, training,
and executing computational networks. It is also a framework for describing
arbitrary learning machines such as deep neural networks (DNNs). CNTK is an
implementation of computational networks that supports both CPU and GPU.
 
This page describes the Python API for CNTK_ version 2.0rc1. This is an ongoing effort
to expose such an API to the CNTK system, thus enabling the use of higher-level
tools such as IDEs to facilitate the definition of computational networks, to execute
them on sample data in real time. Please give feedback through these `channels`_.

We have a new type system in the layers module to make the input type more readable.
This new type system is subject to change, please give us feedback on github or stackoverflow

.. toctree::
   :maxdepth: 2

   Setup <setup>
   Getting Started <gettingstarted>
   Working with Sequences <sequence>
   Tutorials <tutorials>
   Examples <examples>
   Layers Library Reference  <layerref>
   Python API Reference <apireference>
   Readers, Multi-GPU, Profiling...<readersprofetc>
   Extending CNTK <extend>
   Known Issues <knownissues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _`channels`: https://github.com/Microsoft/CNTK/wiki/Feedback-Channels
