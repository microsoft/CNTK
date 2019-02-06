import cntk as C
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components

input_vocab_dim  = 165393
label_vocab_dim  = 165393

def create_reader(path, is_training):
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
        features = C.io.StreamDef(field='M', shape=input_vocab_dim, is_sparse=True),
        labels   = C.io.StreamDef(field='R', shape=label_vocab_dim, is_sparse=True)
    )), randomize = True, max_sweeps = 1, multithreaded_deserializer=True)

# Train data reader
reader = create_reader('top_50.ctf', True)

# Source and target inputs to the model
inputAxis = C.Axis('inputAxis')
labelAxis = C.Axis('labelAxis')
InputSequence = C.layers.SequenceOver[inputAxis]
LabelSequence = C.layers.SequenceOver[labelAxis]

def batch_only(reader, max_epochs, epoch_size):

    # Instantiate the trainer object to drive the model training
    minibatch_size = 2048
    # Get minibatches of sequences to train with and perform model training
    total_samples = 0

    for epoch in range(max_epochs):
        while total_samples < (epoch+1) * epoch_size:
            # get next minibatch of training data
            mb_train = reader.next_minibatch(minibatch_size)

            total_samples += mb_train[reader.streams.labels].num_samples

    print("%d epochs complete." % max_epochs)

batch_only(reader, max_epochs=1, epoch_size=5662308)
