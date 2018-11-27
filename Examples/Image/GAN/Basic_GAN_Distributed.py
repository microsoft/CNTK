import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

import cntk as C
import cntk.tests.test_utils
from timeit import default_timer as timer

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

#%matplotlib inline
isFast = True 
# architectural parameters
g_input_dim = 100
g_hidden_dim = 128
g_output_dim = d_input_dim = 784
d_hidden_dim = 128
d_output_dim = 1

# Ensure the training data is generated and available for this tutorial
def create_reader(path, is_training, input_dim, label_dim):
    deserializer = C.io.CTFDeserializer(
        filename = path,
        streams = C.io.StreamDefs(
            labels_unused = C.io.StreamDef(field = 'labels', shape = label_dim, is_sparse = False),
            features = C.io.StreamDef(field = 'features', shape = input_dim, is_sparse = False
            )
        )
    )
    return C.io.MinibatchSource(
        deserializers = deserializer,
        randomize = is_training,
        max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1
    )

np.random.seed(123)
def noise_sample(num_samples):
    return np.random.uniform(
        low = -1.0,
        high = 1.0,
        size = [num_samples, g_input_dim]
    ).astype(np.float32)

def generator(z):
    with C.layers.default_options(init = C.xavier()):
        h1 = C.layers.Dense(g_hidden_dim, activation = C.relu)(z)
        return C.layers.Dense(g_output_dim, activation = C.tanh)(h1)

def discriminator(x):
    with C.layers.default_options(init = C.xavier()):
        h1 = C.layers.Dense(d_hidden_dim, activation = C.relu)(x)
        return C.layers.Dense(d_output_dim, activation = C.sigmoid)(h1)

# training config
minibatch_size = 1024
num_minibatches = 300 if isFast else 40000
lr = 0.00005

def build_graph(noise_shape, image_shape, G_progress_printer, D_progress_printer):
    input_dynamic_axes = [C.Axis.default_batch_axis()]
    Z = C.input_variable(noise_shape, dynamic_axes=input_dynamic_axes)
    X_real = C.input_variable(image_shape, dynamic_axes=input_dynamic_axes)
    X_real_scaled = 2*(X_real / 255.0) - 1.0

    # Create the model function for the generator and discriminator models
    X_fake = generator(Z)
    D_real = discriminator(X_real_scaled)
    D_fake = D_real.clone(
        method = 'share',
        substitutions = {X_real_scaled.output: X_fake.output}
    )

    # Create loss functions and configure optimazation algorithms
    G_loss = 1.0 - C.log(D_fake)
    D_loss = -(C.log(D_real) + C.log(1.0 - D_fake))

    G_learner = C.fsadagrad(
       parameters = X_fake.parameters,
        lr = C.learning_parameter_schedule_per_sample(lr),
        momentum = C.momentum_schedule_per_sample(0.9985724484938566)
    )
    D_learner = C.fsadagrad(
        parameters = D_real.parameters,
        lr = C.learning_parameter_schedule_per_sample(lr),
        momentum = C.momentum_schedule_per_sample(0.9985724484938566)
    )

    DistG_learner = C.train.distributed.data_parallel_distributed_learner(G_learner)
    
    # The following API marks a learner as the matric aggregator, which is used by 
    # the trainer to determine the training progress.
    # It is required, only when more than one learner is provided to a *single* trainer. 
    # In this example, we use two trainers each with a single learner, so it 
    # is not required and automatically set by CNTK for each single learner. However, if you 
    # plan to use both learners with a single trainer, then it needs to be call before 
    # creating the trainer.
    #DistG_learner.set_as_metric_aggregator()

    DistD_learner = C.train.distributed.data_parallel_distributed_learner(D_learner)

    # Instantiate the trainers
    G_trainer = C.Trainer(
        X_fake,
        (G_loss, None),
        DistG_learner,
        G_progress_printer
    )
    D_trainer = C.Trainer(
        D_real,
        (D_loss, None),
        DistD_learner,
        D_progress_printer
    )

    return X_real, X_fake, Z, G_trainer, D_trainer

def train(reader_train):
    k = 2
    worker_rank = C.Communicator.rank()
    # print out loss for each model for upto 50 times
    print_frequency_mbsize = num_minibatches // 50
    pp_G = C.logging.ProgressPrinter(print_frequency_mbsize, rank=worker_rank)
    pp_D = C.logging.ProgressPrinter(print_frequency_mbsize * k, rank=worker_rank)

    X_real, X_fake, Z, G_trainer, D_trainer = \
        build_graph(g_input_dim, d_input_dim, pp_G, pp_D)
    
    input_map = {X_real: reader_train.streams.features}

    num_partitions = C.Communicator.num_workers()
    worker_rank = C.Communicator.rank()
    distributed_minibatch_size  = minibatch_size // num_partitions

    for train_step in range(num_minibatches):

        # train the discriminator model for k steps
        for gen_train_step in range(k):
            Z_data = noise_sample(distributed_minibatch_size)
            X_data = reader_train.next_minibatch(minibatch_size, input_map, num_data_partitions=num_partitions, partition_index=worker_rank)

            if X_data[X_real].num_samples == Z_data.shape[0]:
                batch_inputs = {X_real: X_data[X_real].data, 
                                Z: Z_data}
                D_trainer.train_minibatch(batch_inputs)

        # train the generator model for a single step
        Z_data = noise_sample(distributed_minibatch_size)
        batch_inputs = {Z: Z_data}
        G_trainer.train_minibatch(batch_inputs)

        G_trainer_loss = G_trainer.previous_minibatch_loss_average

    return Z, X_fake, G_trainer_loss

def plot_images(images, subplot_shape):
    plt.style.use('ggplot')
    fig, axes = plt.subplots(*subplot_shape)
    for image, ax in zip(images, axes.flatten()):
        ax.imshow(image.reshape(28, 28), vmin = 0, vmax = 1.0, cmap = 'gray')
        ax.axis('off')
    plt.show()
    

#mpiexec entrance
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-datadir', '--datadir')
    args = vars(parser.parse_args())

    data_found = False
    train_file = os.path.join(args['datadir'], "Train-28x28_cntk_text.txt")
    if os.path.isfile(train_file):
        data_found = True
    if not data_found:
        raise ValueError("Please generate the data by completing CNTK 103 Part A")
    
    worker_rank = C.Communicator.rank()
    
    start = timer()
    reader_train = create_reader(train_file, True, d_input_dim, label_dim=10)
    G_input, G_output, G_trainer_loss = train(reader_train)
    # Print the generator loss 
    C.Communicator.finalize()
    end = timer()
    print("Training loss of the generator at worker: {%d} is: {%f}, time taken is: {%d} seconds."%(worker_rank, G_trainer_loss, (end - start)))
    
    # Please uncomment below to display the generated images.
    #if worker_rank == 0:
    #    noise = noise_sample(36)
    #    images = G_output.eval({G_input: noise})
    #    plot_images(images, subplot_shape =[6, 6])
