# CIFAR-10 Dataset

The CIFAR-10 dataset (http://www.cs.toronto.edu/~kriz/cifar.html) is a popular dataset for image classification, collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is a labeled subset of the [80 million tiny images](http://people.csail.mit.edu/torralba/tinyimages/) dataset.

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The 10 classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The CIFAR-10 dataset is not included in the CNTK distribution but can be easily downloaded and converted to CNTK-supported format by cd to this directory, Examples/Image/DataSets/CIFAR-10 and running the following Python command:

```
python install_cifar10.py
```

After running `install_cifar10.py`, you will see the original CIFAR-10 data are copied in a folder named `cifar-10-batches-py`. Meanwhile, two text files `Train_cntk_text.txt` and `Test_cntk_text.txt` are created in the current folder. These text files can be read directly by CNTK.

In addition, the script will create a `train` and a `test` folder that store train and test images in png format. It will also create appropriate mapping files (`train_map.txt` and `test_map.txt`) for the CNTK `ImageReader` as well as mean file `CIFAR-10_mean.xml`.

The total amount of disk space required for both the text version and the png version for CIFAR-10 is around `950`MB. 

We provide multiple examples in the [Classification](../../Classification) folder to train classifiers for CIFAR-10 with CNTK. Please refer there for more details.

If you are curious about how well computers can perform on CIFAR-10 today, Rodrigo Benenson maintains a [blog](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130) on the state-of-the-art performance of various algorithms.
