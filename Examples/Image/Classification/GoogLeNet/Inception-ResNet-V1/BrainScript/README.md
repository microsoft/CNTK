# CNTK Examples: Image/Classification/GoogLeNet/Inception-ResNet-V1

## BrainScript

### Inception-ResNet-V1.cntk

The Inception-ResNet-V1 model is implemented according to the model described in [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261).

This implementation achieves `78.210%` Top-1 accuracy and `93.848%` Top-5 accuracy in validation set.

And we're using common SGD with Nesterov momentum in this implementation.

This example with a `256` batch-size should be trained with 8 GPUs.

You could run the example from the current folder using:

`mpiexec -n 8 cntk configFile=Inception-ResNet-V1.cntk`

If you would like to run this example with a single card. You could divide the `minibatchSize` and `learningRatesPerMB` with a ratio `8` simultaneously.

And run this example using:

`cntk configFile=Inception-ResNet-V1.cntk`
