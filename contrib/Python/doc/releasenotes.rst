Release Notes
=============

Version 1.4 (April 2016)
------------------------

New and improved features:

* Python API: This is the first release containing a limited version of the 
  Python API. It exposes CNTKTextFormatReader, SGD, local and deferred 
  execution, and 22 operators.

* CNTK Core: 

  * This release contains a new generic text format reader called 
    CNTKTextFormatReader. UCIFastReader has been deprecated. The 
    new reader by definition supports all tensor formats (sparse and dense,
    sequences and non-sequences, multiple inputs) that can be fed to CNTK.

  * The concept of named dynamic axes has been exposed in the configuration, 
    which enables modelling of inputs of varying length.

Current restrictions of the Python API:

* Although CNTK implements more than 100 operators through internal APIs, only 
  s small subset have been exposed through the Python API at this
  point. We are using this API as a production gate, requiring unit tests and 
  documentation before new functionality is exposed. More operators will be
  added in the following weeks. In particular, convolution operations and
  reductions are missing at this point.

* The Python API is a pure out-of-process API at this point. This means that
  only methods on the context interact with CNTK directly through command line
  calls. An in-process API with far greater extensibility options is planned
  later in 2016 through the 2.0 release.

* The training loop is monolithic at this point and cannot be 
  broken up into single forward/backward passes. This restriction will be 
  lifted with 2.0 release.

* Although inputs can be sparse, sparse features cannot be fed through the
  Python API at this point for immediate evaluation. They can only be fed 
  through files read through the CNTKTextFormatReader.

* We are only exposing the CNTKTextFormatReader in Python at this point. More 
  data formats (ImageReader, speech formats) will be added in a later release.

* We are not exposing a standard layer collection for LSTMs etc. at this point.
  A first version of this will be added in the next release.

* Tensor shapes are only available after a call to the context methods, which
  run graph validation and tensor inference.

* Only few examples have been translated from the CNTK-internal configuration
  format (NDL) to the Python API. More will be added in the next releases.
  
Current restrictions of CNTK Core:

* A tensor can have only one dynamic axis (the outermost one).

* The support for sparse inputs on the operators is... sparse. 
  Operations might throw NotImplementedExceptions when a sparse tensor is fed.
  The exact level of support will be described in the next release.
  
* The built-in criterion nodes aggregate over the whole minibatch. The SGD 
  algorithm divides value this by the number of samples found on the default
  dynamic axis, not the one that was used as input to the criterion node.
  In the next release, criterion nodes will not aggregate over the dynamic
  axis any longer. This logic is moved to SGD itself.
  
  
Roadmap for Version 1.5
-----------------------

We are planning monthly releases for this API. The items on the agenda for 
May release (due end of May/early June) are:

* Python API: Greatly increased list of operators: Shape operations, 
  elementwise operations, reductions.

* Python API: Support for image and speech readers

* Python API: Support for sparse input tensors instead of NumPy arrays, where
  applicable.

* Python API: First version of a layer API 

* Readers: New speach reader

* Readers: Combination of reader deserializers and transformers

* Core: Profiling support

* Core: More operators planned for core.

