# CNTK Examples: Image/Classification/AlexNet

## BrainScript

### AlexNet_ImageNet.cntk

Our AlexNet model is a slight variation of the Caffe implementation of AlexNet (https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet). We removed model parallelism since nowadays most modern GPUs can hold the model in memory. Our model's accuracy is about `59.9%` for top-1 category and `82.2%` for top-5 categories, using just the center crop. In comparison, the BLVC AlexNet accuracy is `57.1%` for top-1 category and `80.2%` for top-5 categories. You may achieve our accuracy numbers by launching the command:

`cntk configFile=AlexNet_ImageNet.cntk`

We will post the pre-trained model shortly. Please check back this page in a few weeks.
