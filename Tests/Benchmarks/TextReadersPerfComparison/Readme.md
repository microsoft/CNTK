# Preparing the data:

1.  Fetch MNIST training data (see Examples\Image\MNIST\Readme.md)
2.  Run  ln -s ..\..\..\..\Examples\Image\MNIST\Data\Train-28x28.txt uci_data
2a. To increase the input file size (tenfold) run: for i in {1..10};do cat uci_data >> uci_data_copy; done; mv uci_data_copy uci_data
3.  Run python Scripts\convert.py uci_data cntk_text_format_data