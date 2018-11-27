# CaffeConverter

The tool will help you convert trained models from Caffe to CNTK.

*Convert trained models:* giving a model script and its weights file, export to CNTK model.

## Dependency

1. Protobuf

    Using `pip install protobuf`

2. Caffe runtime support (Optional)

    While loading Caffe models, Model2CNTK will first try to find Caffe packages in you Python dirs. 
    If failing, system will turn to use protobuf to load the models (maybe long as a few mins). A 
    reference to get Caffe python support:
    
    https://github.com/BVLC/caffe/tree/windows 

3. You may need to compile caffe_pb2.py with following steps:

    a. download and install *`protoc`* in [official website](https://developers.google.com/protocol-buffers)

    b. *protoc -I=$Caffe_DIR --python_out=$DST_DIR $Caffe_DIR/proto/caffe.proto*

    c. copy *caffe_pb2.py* to adapter/bvlccaffe

## Usage and Configuration

`CaffeConverter.from_model(global_conf_path)`

*Usage example:* see [here](./examples/run_convert.py)

*About global conf file:* see guideline in [here](./utils/README.md) and template in [here](./examples/Classification/AlexNet_ImageNet/global.json)

## Support layers

Convolution, Dense/Linear, ReLU, Max/Average Pooling, Softmax, Batch Normalization, LRN, Splice, Plus

## Known Issues

*Model version:* we only support inputs with definition of Input layer. To adapt it, please 
first upgrade your model and prototxt in Caffe tools with commands:

*upgrade_net_proto_text/binary.sh*

## Examples

Please find an example of converter usage in [here](./examples/README.md)