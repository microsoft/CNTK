
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import gzip
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import shutil
import struct

import cntk as C
import cntk.tests.test_utils

from cntk.debugging import start_profiler, stop_profiler, enable_profiler

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#%matplotlib inline



# Functions to load MNIST images and unpack into train and test set.
# - loadData reads a image and formats it into a 28x28 long array
# - loadLabels reads the corresponding label data, one for each image
# - load packs the downloaded image and label data into a combined format to be read later by
#   the CNTK text reader
def download_data():
    def loadData(src, cimg):
        print ('Downloading ' + src)
        gzfname, h = urlretrieve(src, './delete.me')
        print ('Done.')
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack('I', gz.read(4))
                # Read magic number.
                if n[0] != 0x3080000:
                    raise Exception('Invalid file: unexpected magic number.')
                # Read number of entries.
                n = struct.unpack('>I', gz.read(4))[0]
                if n != cimg:
                    raise Exception('Invalid file: expected {0} entries.'.format(cimg))
                crow = struct.unpack('>I', gz.read(4))[0]
                ccol = struct.unpack('>I', gz.read(4))[0]
                if crow != 28 or ccol != 28:
                    raise Exception('Invalid file: expected 28 rows/cols per image.')
                # Read data.
                res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((cimg, crow * ccol))

    def loadLabels(src, cimg):
        print ('Downloading ' + src)
        gzfname, h = urlretrieve(src, './delete.me')
        print ('Done.')
        try:
            with gzip.open(gzfname) as gz:
                n = struct.unpack('I', gz.read(4))
                # Read magic number.
                if n[0] != 0x1080000:
                    raise Exception('Invalid file: unexpected magic number.')
                # Read number of entries.
                n = struct.unpack('>I', gz.read(4))
                if n[0] != cimg:
                    raise Exception('Invalid file: expected {0} rows.'.format(cimg))
                # Read labels.
                res = np.fromstring(gz.read(cimg), dtype = np.uint8)
        finally:
            os.remove(gzfname)
        return res.reshape((cimg, 1))

    def try_download(dataSrc, labelsSrc, cimg):
        data = loadData(dataSrc, cimg)
        labels = loadLabels(labelsSrc, cimg)
        return np.hstack((data, labels))


    # URLs for the train image and label data
    url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    url_train_labels = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    num_train_samples = 60000

    print("Downloading train data")
    train = try_download(url_train_image, url_train_labels, num_train_samples)

    # URLs for the test image and label data
    url_test_image = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    url_test_labels = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    num_test_samples = 10000

    print("Downloading test data")
    test = try_download(url_test_image, url_test_labels, num_test_samples)

    # Save the data files into a format compatible with CNTK text reader
    def savetxt(filename, ndarray):
        dir = os.path.dirname(filename)

        if not os.path.exists(dir):
            os.makedirs(dir)

        if not os.path.isfile(filename):
            print("Saving", filename )
            with open(filename, 'w') as f:
                labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
                for row in ndarray:
                    row_str = row.astype(str)
                    label_str = labels[row[-1]]
                    feature_str = ' '.join(row_str[:-1])
                    f.write('|labels {} |features {}\n'.format(label_str, feature_str))
        else:
            print("File already exists", filename)

    # Save the train and test files (prefer our default path for the data)
    data_dir = os.path.join("..", "Examples", "Image", "DataSets", "MNIST")
    if not os.path.exists(data_dir):
        data_dir = os.path.join("data", "MNIST")

    print ('Writing train text file...')
    savetxt(os.path.join(data_dir, "Train-28x28_cntk_text.txt"), train)

    print ('Writing test text file...')
    savetxt(os.path.join(data_dir, "Test-28x28_cntk_text.txt"), test)

    print('Done')



# Define the data dimensions
input_dim_model = (1, 28, 28)    # images are 28 x 28 with 1 channel of color (gray)
input_dim = 28*28                # used by readers to treat input data as a vector
num_output_classes = 10

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

    ctf = C.io.CTFDeserializer(path, C.io.StreamDefs(
          labels=C.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False),
          features=C.io.StreamDef(field='features', shape=input_dim, is_sparse=False)))

    return C.io.MinibatchSource(ctf,
        randomize = is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)


# Ensure the training and test data is available for this tutorial.
# We search in two locations in the toolkit for the cached MNIST data set.

data_found=False # A flag to indicate if train/test data found in local cache
for data_dir in [os.path.join("..", "Examples", "Image", "DataSets", "MNIST"),
                 os.path.join("data", "MNIST")]:

    train_file=os.path.join(data_dir, "Train-28x28_cntk_text.txt")
    test_file=os.path.join(data_dir, "Test-28x28_cntk_text.txt")

    if os.path.isfile(train_file) and os.path.isfile(test_file):
        data_found=True
        break

if not data_found:
    download_data()
    #raise ValueError("Please generate the data by completing CNTK 103 Part A")

print("Data directory is {0}".format(data_dir))


x = C.input_variable(input_dim_model)
y = C.input_variable(num_output_classes)


# function to build model

def create_model(features):
    with C.layers.default_options(init=C.glorot_uniform(), activation=C.relu):
            h = features
            h = C.layers.Convolution(filter_shape=(5,5),
                                       num_filters=8,
                                       strides=(2,2),
                                       pad=True, name='first_conv')(h)
            print(h.W.shape)
            print(h.b.shape)
            h = C.layers.Convolution2D(filter_shape=(5,5),
                                       num_filters=16,
                                       strides=(2,2),
                                       pad=True, name='second_conv')(h)
            r = C.layers.Dense(num_output_classes, activation=None, name='classify')(h)
            return r


# Create the model
z = create_model(x)

# Print the output shapes / parameters of different components
print("Output Shape of the first convolution layer:", z.first_conv.shape)
print('Output Shape of the second convolution layer:', z.second_conv.shape)
print("Bias value of the last dense layer:", z.classify.b.value)


# Number of parameters in the network
C.logging.log_number_of_parameters(z)


def create_criterion_function(model, labels):
    loss = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error(model, labels)
    return loss, errs # (model, labels) -> (loss, error metric)

# Define a utility function to compute the moving average sum.
# A more efficient implementation is possible with np.cumsum() function
def moving_average(a, w=5):
    if len(a) < w:
        return a[:]    # Need to send a copy of the array
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


# Defines a utility that prints the training progress
def print_training_progress(trainer, mb, frequency, verbose=1):
    training_loss = "NA"
    eval_error = "NA"

    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print ("Minibatch: {0}, Loss: {1:.4f}, Error: {2:.2f}%".format(mb, training_loss, eval_error*100))

    return mb, training_loss, eval_error


def train_test(train_reader, test_reader, model_func, num_sweeps_to_train_with=10):

    # Instantiate the model function; x is the input (feature) variable
    # We will scale the input image pixels within 0-1 range by dividing all input value by 255.
    model = model_func(x/255)

    # Instantiate the loss and error function
    loss, label_error = create_criterion_function(model, y)

    # Instantiate the trainer object to drive the model training
    learning_rate = 0.2
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, label_error), [learner])

    # Initialize the parameters for the trainer
    minibatch_size = 64
    num_samples_per_sweep = 60000  # 60000
    num_minibatches_to_train = (num_samples_per_sweep * num_sweeps_to_train_with) / minibatch_size

    # Map the data streams to the input and labels.
    input_map={
        y  : train_reader.streams.labels,
        x  : train_reader.streams.features
    }

    # Uncomment below for more detailed logging
    training_progress_output_freq = 20

    # Start a timer
    start = time.time()

    C.debugging.debug.set_node_timing(True)

    for i in range(0, int(num_minibatches_to_train)):
        # Read a mini batch from the training data file
        data=train_reader.next_minibatch(minibatch_size, input_map=input_map)

        trainer.train_minibatch(data)
        print_training_progress(trainer, i, training_progress_output_freq, verbose=1)

    # Print training time
    print("Training took {:.1f} sec".format(time.time() - start))
    print(trainer.print_node_timing())
    input('pause')

    # Test the model
    test_input_map = {
        y  : test_reader.streams.labels,
        x  : test_reader.streams.features
    }

    # Test data for trained model
    test_minibatch_size = 512
    num_samples = 10000
    num_minibatches_to_test = num_samples // test_minibatch_size

    test_result = 0.0

    for i in range(num_minibatches_to_test):

        # We are loading test data in batches specified by test_minibatch_size
        # Each data point in the minibatch is a MNIST digit image of 784 dimensions
        # with one pixel per dimension that we will encode / decode with the
        # trained model.
        data = test_reader.next_minibatch(test_minibatch_size, input_map=test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error

    # Average of evaluation errors of all test minibatches
    print("Average test error: {0:.2f}%".format(test_result*100 / num_minibatches_to_test))


def do_train_test():
    global z
    z = create_model(x)
    reader_train = create_reader(train_file, True, input_dim, num_output_classes)
    reader_test = create_reader(test_file, False, input_dim, num_output_classes)

    start_profiler()

    enable_profiler()

    train_test(reader_train, reader_test, z)

    stop_profiler()

do_train_test()


out = C.softmax(z)

# Read the data for evaluation
reader_eval=create_reader(test_file, False, input_dim, num_output_classes)

eval_minibatch_size = 25
eval_input_map = {x: reader_eval.streams.features, y:reader_eval.streams.labels}

data = reader_eval.next_minibatch(eval_minibatch_size, input_map=eval_input_map)

img_label = data[y].asarray()
img_data = data[x].asarray()

# reshape img_data to: M x 1 x 28 x 28 to be compatible with model
img_data = np.reshape(img_data, (eval_minibatch_size, 1, 28, 28))

predicted_label_prob = [out.eval(img_data[i]) for i in range(len(img_data))]


# Find the index with the maximum value for both predicted as well as the ground truth
pred = [np.argmax(predicted_label_prob[i]) for i in range(len(predicted_label_prob))]
gtlabel = [np.argmax(img_label[i]) for i in range(len(img_label))]


print("Label    :", gtlabel[:25])
print("Predicted:", pred)


sample_number = 5
plt.imshow(img_data[sample_number].reshape(28,28), cmap="gray_r")
plt.axis('off')

img_gt, img_pred = gtlabel[sample_number], pred[sample_number]
print("Image Label: ", img_pred)


input('complete')