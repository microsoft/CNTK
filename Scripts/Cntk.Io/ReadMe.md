# CNTK.IO, a I/O library for CNTK

CNTK is the open source deep learning toolkit of Microsoft.
This library is intended as a high level .NET I/O library
to facilitate moving data from .NET to CNTK and vice-versa.

## CNTKDataBinaryWriter

The `CNTKDataBinaryWriter` is intended for serializing datasets.
It adopts the recommended generic binary format of CNTK which is
a columnar chunked data format.

Terminology:

* **Input streams**: the "columns" of a kind present in the dataset.
* **Sequences**: the individual instances - a very CTNK-specific terminology.
* **Samples**: the count of non-zero values within each sequence.

Limitations:

* Only `float` precision supported
* Only single sequence input streams supported

Further improvements (being considered):

* auto-chunking based on target MB size for chunks
* introducing a 'reader' counterpart (and unit tests)
* supporting `double`

