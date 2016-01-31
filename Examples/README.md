# CNTK Examples

This folder contains demos and examples to get started with CNTK.
The examples are structured by topic into Image, Speech, Text and Other.
The individual folders contain on the first level at least one self-contained example,
which cover different types of networks including FF, CNN, RNN and LSTM.
Further examples for each category are provided in the corresponding Miscellaneous subfolder.
Each folder contains a Readme file that explains how to run the example on Windows and Linux. 
How to run the examples on Philly (https://philly) is explained in the Philly portal wiki 
(Philly is an internal GPU cluster for Microsoft production runs).

The examples shown in the table below provide a good introduction to CNTK.
Please refer to the Readme file in the corresponding folder for further details.

|Folder                   | Domain                                           | Network types   |
|:------------------------|:-------------------------------------------------|:----------------|
|Other/Simple2d           | Synthetic 2d data                                | FF (CPU and GPU)
|Speech/AN4               | Speech data (CMU AN4)                            | FF and LSTM
|Image/MNIST              | Image data (MNIST handwritten digit recognition) | CNN 
|Text/PennTreebank        | Text data (penn treebank)                        | RNN
