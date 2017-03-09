This directory contains different scripts to support CNTK.

## CNTK Binary Installers

The directory `install` contains scripts which are used in the CNTK binary download to install 
CNTK on a users system. They are NOT intended to run from this location in the repository.

* `install/windows` - A script for installing a Windows CNTK *binary* drop, cf. [here](https://github.com/Microsoft/CNTK/wiki/Setup-Windows-Binary-Script).
* `install/linux` - A script for installing a Linux CNTK *binary* drop, cf. [here](https://github.com/Microsoft/CNTK/wiki/Setup-Linux-Binary-Script).

## CNTK Text format Converters

Two Python Scripts for converting Data to CNTK Text format for using as an input for CNTK Text Format 
Reader (see https://github.com/microsoft/CNTK/wiki/CNTKTextFormat-Reader).

### Convert Dictionary to Text

`txt2ctf.py` converts a set of dictionary files and a plain text file to CNTK Text format.

Run `python txt2ctf.py -h` to see usage instructions. See the comments in the beginning of the script 
file for the specific usage example. 

### Convert UCI Format to Text

`uci2ctf.py` converts data stored in a text file in UCI format to CNTK Text format. 

Run `python uci2ctf.py -h` to see usage instructions and example. 

For Example:
```
python Scripts/uci2ctf.py --input_file Examples/Image/MNIST/Data/Train-28x28.txt --features_start 1 --features_dim 784 --labels_start 0 --labels_dim 1 --num_labels 10  --output_file Examples/Image/MNIST/Data/Train-28x28_cntk_text.txt
```
- `input_file` - original dataset in the (columnar) UCI format
- `features_start` - index of the first feature column (start parameter in the UCIFastReader config, see [here](https://github.com/Microsoft/CNTK/wiki/UCI-Fast-Reader)
- `features_dim` - number of feature columns (dim parameter in the UCIFastReader config)
- `labels_start` - index of the first label column
- `labels_dim` - number of label columns
- `num_labels` - number of possible label values (labelDim parameter in the UCIFastReader config)
- `output_file` - path and filename of the resulting dataset.

