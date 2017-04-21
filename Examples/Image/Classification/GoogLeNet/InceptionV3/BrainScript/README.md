# CNTK Examples: Image/Classification/GoogLeNet/InceptionV3

## BrainScript

### InceptionV3.cntk

The Inception-V3 model is implemented according to the model described in [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) with changes from TensorFlow implementation.

This implementation achieves `77.460%` Top-1 accuracy and `93.562%` Top-5 accuracy in validation set, which is comparable to TensorFlow implementation.

This example with a `256` batch-size should be trained with 8 GPUs. 

You could run the example from the current folder using:

`mpiexec -n 8 cntk configFile=InceptionV3.cntk`

If you would like to run this example with a single card. You could divide the `minibatchSize` and `learningRatesPerMB` with a ratio `8` simultaneously.

And run this example using:

`cntk configFile=InceptionV3.cntk`
