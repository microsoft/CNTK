# CNTK Demos and Example Setups

This folder contains a few self-contained demos to get started with CNTK.
The data for the demos is contained in the corresponding Data folders.
Each demo folder has a Readme file that explains how to run it on Windows and Linux. 
How to run the demos on Philly (https://phillywebportal.azurewebsites.net/index.aspx) is 
explained in the Philly portal wiki (Philly is an internal GPU cluster for Microsoft production runs).
The demos cover different domains such as Speech and Text classification 
and show different types of networks including FF, CNN RNN and LSTM.

Further examples are provided in the folder 'ExampleSetups'. 
A popular example is the MNIST handwritten digits classification task. 
You can find this example in 'ExampleSetups/Images/MNIST'.
The examples in 'ExampleSetups' might require downloading data which in some cases is not free of charge. 
See individual folders for more information.

The four examples shown in the table below provide a good introduction to CNTK.
Additional more complex examples can be found in the 'ExampleSetups' folder.

|Folder                   | Domain                                           | Network types   |
|:------------------------|:-------------------------------------------------|:----------------|
Demos/Simple2d            | Synthetic 2d data                                | FF (CPU and GPU)
Demos/Speech              | Speech data (CMU AN4)                            | FF and LSTM
Demos/Text                | Text data (penn treebank)                        | RNN
ExampleSetups/Image/MNIST | Image data (MNIST handwritten digit recognition) | CNN 
