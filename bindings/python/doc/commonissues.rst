This is an alpha release meant for early users to try the bits and
provide feedback on the usability and functional aspects of the API.
These bits have undergone limited testing so far, so expect some rough
edges. **Also please expect the API to undergo changes over the coming
weeks, which may break backwards compatibility of programs written
against the alpha release**.

-  Only a subset of the planned functionality is available; features
   like distributed training, automatic LR and MB size search and API
   extensibility will become available over the next few weeks.

-  Python 2.7 support is currently unavailable but will be part of the
   upcoming beta release.

-  On Windows only Python 3.4 is supported and not Python 3.5 since the
   latter requires Visual Studio 2015 which CNTK has not yet migrated
   to. This will also be addressed before the upcoming beta release.

-  The core API itself is implemented in C++ for speed and efficiency
   and python bindings are created through SWIG. We are increasingly
   creating thin python wrappers for the APIs to attach docstrings to,
   but this is a work in progress and for some of the APIs, you may
   directly encounter SWIG generated API definitions (which are not the
   prettiest to read).

-  Shape and dimension inference support is currently unavailable and
   the shapes of all Variable objects have to be fully specified.
