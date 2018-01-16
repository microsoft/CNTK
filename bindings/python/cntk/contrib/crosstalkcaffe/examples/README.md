# Examples

## Run examples:

1. Prepare the CNTK python environment.

2. Download the model weights of target models and move them into the same folder with global.json, respectively.

3. Run the command:

*<center>python run_convert.py [`Parameter`, default=ALL]</center>*

4. Support models (target by *`Parameter`*):
AlexNet, NIN, VGG16, VGG19, GoogLeNet, ResNet50, ResNet101, ResNet152, ALL

## Model weights links:

*VGG*: http://www.robots.ox.ac.uk/~vgg/research/very_deep

*ResNet*: https://github.com/KaimingHe/deep-residual-networks

*BVLC-GoogLeNet*: http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel

*AlexNet*: http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel

*NIN*: https://gist.github.com/mavenlin/d802a5849de39225bcc6

*For some early models, using `upgrade_net_proto_binary` tool of Caffe to convert the legacy model to recent formats.*

## Results

*ImageNet*

| Network | Baseline(top1/top5) |    Converted(top1/top5)    | Link                  |
|:--------------:|:---------:|:---------:|:---------:|
| VGG16_ImageNet | -/7.5% | -/10.13% | [VGG_16_ImageNet](./Classification/VGG_ImageNet) |
| VGG19_ImageNet | -/7.5% | -/10.20% | [VGG_19_ImageNet](./Classification/VGG_ImageNet) |
| ResNet50_ImageNet | 24.7%/7.8% | 24.87%/7.76% | [ResNet_50_ImageNet](./Classification/ResNet_ImageNet) |
| ResNet101_ImageNet | 23.6%/7.1% | 23.56%/7.12% | [ResNet_101_ImageNet](./Classification/ResNet_ImageNet) |
| ResNet152_ImageNet | 23.0%/6.7% | 23.34%/6.72% | [ResNet_152_ImageNet](./Classification/ResNet_ImageNet) |
| GoogLeNet_ImageNet | 31.3%/11.1% | 33.43%/11.54% | [GoogLeNet_ImageNet](./Classification/GoogLeNet_ImageNet) |
| NIN_ImageNet | 40.61%/- | 45.79%/19.8% | [NIN_ImageNet](./Classification/NIN_ImageNet) |

*'-' means the results are not reported.*