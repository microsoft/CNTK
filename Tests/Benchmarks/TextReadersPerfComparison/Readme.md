# Preparing the data:

1.  Fetch MNIST training data (see Examples\Image\MNIST\Readme.md)
2.  Run  ln -s ..\..\..\..\Examples\Image\MNIST\Data\Train-28x28.txt uci_data
2a. To increase the input file size (tenfold) run: for i in {1..10};do cat uci_data >> uci_data_copy; done; mv uci_data_copy uci_data
3.  Run python Scripts\convert.py uci_data cntk_text_format_data


> x64\Release_CpuOnly\TextReadersPerfComparison.exe e:\data\cntk_text_experiments\benchmarks\
Building an input data index.
UCIFastReader : 594 ms
CNTKTextFormatReader: 141ms
Parsing input data (single pass).
UCIFastReader : 1437 ms
CNTKTextFormatReader : 1625ms
Reading 150000 sequences.
UCIFastReader : 3562 ms
CNTKTextFormatReader : 3750ms
Reading 600000 sequences.
UCIFastReader : 17828 ms
CNTKTextFormatReader : 17578ms
Reading 1000000 sequences.
UCIFastReader : 29047 ms
CNTKTextFormatReader : 18391ms