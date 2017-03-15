# CNTK Examples: Image/Classification/GoogLeNet/BN-Inception

## BrainScript

### BN-Inception.cntk

The BN-Inception model is implemented according to the model described in [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).

This implementation achieves `74.938%` Top-1 accuracy and `92.346%` Top-5 accuracy in our test, which is slightly better than the result in Google’s original paper – `74.8%` Top-1 accuracy.

This example with a `256` batch-size should be trained with 8 GPUs. 

You could run the example from the current folder using:

`mpiexec -n 8 cntk configFile=BN-Inception.cntk`

If you would like to run this example with a single card. You could divide the `minibatchSize` and `learningRatesPerMB` with a ratio `8` simultaneously.

And run this example using:

`cntk configFile=BN-Inception.cntk`