Multiverso
==========
[![Build Status](https://travis-ci.org/Microsoft/Multiverso.svg?branch=master)](https://travis-ci.org/Microsoft/Multiverso)

Multiverso is a parameter server based framework for training machine learning models on big data with numbers of machines. It is currently a standard C++ library and provides a series of friendly programming interfaces, and it is extended to support calling from python and Lua programs. With such easy-to-use APIs, machine learning researchers and practitioners do not need to worry about the system routine issues such as distributed model storage and operation, inter-process and inter-thread communication, multi-threading management, and so on.
Instead, they are able to focus on the core machine learning logics: data, model, and training.

For more details, please view our website [http://www.dmtk.io](http://www.dmtk.io).

Build
----------

**Linux** (Tested on Ubuntu 14.04)

```
sudo apt-get install libopenmpi-dev openmpi-bin build-essential cmake git
git clone https://github.com/Microsoft/multiverso.git && cd multiverso
mkdir build && cd build
cmake .. && make && sudo make install
```

**Windows**

Open the `Multiverso.sln` with [Visual Studio 2013]() and build.

Related Projects
----------

Current distributed systems based on multiverso:

* [lightLDA](https://github.com/Microsoft/lightlda): Scalable, fast, lightweight system for large scale topic modeling
* [distributed_word_embedding](https://github.com/Microsoft/multiverso/tree/master/Applications/WordEmbedding) Distributed system for word embedding
* [distributed_word_embedding(deprecated)](https://github.com/Microsoft/distributed_word_embedding) Distributed system for word embedding
* [distributed_skipgram_mixture(deprecated)](https://github.com/Microsoft/distributed_skipgram_mixture) Distributed skipgram mixture for multi-sense word embedding

Microsoft Open Source Code of Conduct
------------

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
