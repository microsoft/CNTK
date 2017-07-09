# Model2CNTK

The project will help you convert training scripts or trained models from Caffe to CNTK.

*Convert training scripts:* updating, coming soon

*Convert trained models:* giving a model script and its weights file, export to CNTK model.

## Environment

We setup the local environment with `Windows 10, Python 2.7/3.5`.

## Dependency

1. Protobuf

    Using `pip install protobuf`

3. Caffe runtime support (Optional)

    While loading Caffe models, Model2CNTK will first try to find Caffe packages in you Python dirs. 
    If failing, system will turn to use protobuf to load the models (maybe long as a few mins). A 
    reference to get Caffe python support:
    
    https://github.com/BVLC/caffe/tree/windows 

## Command and Global Configuration

python cntkboard.py --option *[convert_script/convert_model]* --conf *[global_conf_file]*

*About global conf file:* find template settings in [Example](./examples/Classification/AlexNet_ImageNet/global.json)

## Support layers

Convolution, Dense/Linear, ReLU, Max/Average Pooling, Softmax, Batch Normalization, LRN, Splice, Plus

## Known Issues

*Model version:* we only support inputs with definition of Input layer. To adapt it, please 
first upgrade your model and prototxt with *upgrade_net_proto_text/binary.sh* (Caffe tools).

## Examples

*Classification:*

1. VGG-Net: [Link](./examples/Classification/VGG_ImageNet)

2. ResNet: [Link](./examples/Classification/ResNet_ImageNet)

3. GoogLeNet: [Link](./examples/Classification/GoogLeNet_ImageNet)

4. NIN: [Link](./examples/Classification/NIN_ImageNet)

5. AlexNet: [Link](./examples/Classification/AlexNet_ImageNet)

## Help 

Please feel free to ping [yuxiao guo](v-yuxgu@microsoft.com)