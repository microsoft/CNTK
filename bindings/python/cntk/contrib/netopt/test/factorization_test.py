import numpy as np
import pytest
import cntk as C
import cntk.contrib.netopt.factorization as nc
C.cntk_py.set_fixed_random_seed(1)
C.cntk_py.force_deterministic_algorithms()

# create a dense network for the tests
def _create_model_dense(features, num_hidden_layers, hidden_layers_dim, num_output_classes):
    with C.layers.default_options(init=C.layers.glorot_uniform(), activation=C.sigmoid):
        h = features
        for _ in range(num_hidden_layers):
            h = C.layers.Dense(hidden_layers_dim)(h)
        last_layer = C.layers.Dense(num_output_classes, activation = None)

        return last_layer(h)


# no size reduction, only the factorization.
def _get_rank_same_size(W):
        return int(len(W) * 1)


# reduce the size by len* 0.8
def _get_rank_reduced_size(W):
    return int(len(W) * 0.8)
 
   
# filter dense blocks that has the same height and width.
def _filter(model):
    W = model.W.value
    if (len(W) != len(W[0])):
        return False
    else:
        return True


# Helper function to generate a random data sample
def _generate_random_data_sample(sample_size, feature_dim, num_classes):
    Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)
    X = (np.random.randn(sample_size, feature_dim)+3) * (Y+1)
    X = X.astype(np.float32)
    class_ind = [Y==class_number for class_number in range(num_classes)]
    Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    return X, Y


def test_svd_factorization():
    # W and its svd factorizations (U and sV)
    W = np.array([[1, 0, 0, 0, 2], 
         [0, 0, 3, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 2, 0]])
    
    U = np.array([[0, 1, 0, 0],
           [1, 0, 0, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 0]])
    
    sV = np.array([[0, 0, 3, 0, 0], 
           [1, 0, 0, 0, 2],
           [0, 0, 0, 2, 0],
           [0, 0, 0, 0, 0]])
    
    # call svd factorization with W's length
    W1, W2 = nc.svd_subprojection(W, len(W))
   
    assert(np.array_equal(W1, U) == True)   
    assert(np.allclose(sV, W2) == True)


def test_factor_dense():

    input_dim = 2
    num_output_classes = 2
    hidden_layer_dim = 50

    input = C.input_variable(input_dim)
    z = _create_model_dense(input, input_dim, hidden_layer_dim, num_output_classes)
    
    newz = nc.factor_dense(z, projection_function=_get_rank_same_size, filter_function = _filter)
    newblocks = C.logging.graph.depth_first_search(
                    newz, lambda x : type(x) == C.Function and x.root_function.is_block, depth = 0)
    
    assert(newblocks[1].op_name == "DenseFactored")    
    block_root = C.as_composite(newblocks[1].block_root)
    # no reduction, same size but factored.
    assert(block_root.W1.value.shape == (50, 50))
    
    newz = nc.factor_dense(z, projection_function=_get_rank_reduced_size, filter_function = _filter)
    newblocks = C.logging.graph.depth_first_search(
                    newz, lambda x : type(x) == C.Function and x.root_function.is_block, depth = 0)
    assert(newblocks[1].op_name == "DenseFactored")    
    block_root = C.as_composite(newblocks[1].block_root)
    # the reduction has taken place now.
    assert(block_root.W1.value.shape == (50, 40))


def _percentage_match(labels, predictions):
    match_count = 0
    for idx, lbl in enumerate(labels): 
        if (np.argmax(lbl) == np.argmax(predictions[idx])):
            match_count += 1
    return match_count / len(labels) * 100 if len(labels) != 0  else 0


def test_factor_dense_for_prediction():

    input_dim = 2
    num_output_classes = 2
    hidden_layer_dim = 50
    num_minibatches_to_train = 2000
    minibatch_size = 25
    learning_rate = 0.5

    input = C.input_variable(input_dim)
    label = C.input_variable(num_output_classes)

    z = _create_model_dense(input, input_dim, hidden_layer_dim, num_output_classes)

    loss = C.cross_entropy_with_softmax(z, label)
    eval_error = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training

    lr_schedule = C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    learner = C.sgd(z.parameters, lr_schedule)
    trainer = C.Trainer(z, (loss, eval_error), [learner])


    # Run the trainer and perform model training
    training_progress_output_freq = 20
    plotdata = {"batchsize":[], "loss":[], "error":[]}


    for i in range(0, int(num_minibatches_to_train)):
        features, labels = _generate_random_data_sample(minibatch_size, input_dim, num_output_classes)
        # Specify the input variables mapping in the model to actual minibatch data for training
        trainer.train_minibatch({input : features, label : labels})
    
    # generate some data to predict
    features, labels = _generate_random_data_sample(10, 2, 2)

    # factor the model.
    newz = nc.factor_dense(z, projection_function=_get_rank_reduced_size, filter_function = _filter)
    original_out = C.softmax(z)
    factored_out = C.softmax(newz)

    original_labels_probs = original_out.eval({input : features})
    predicted_label_probs = factored_out.eval({input : features})
    
    original_prediction_percentage = _percentage_match(labels, original_labels_probs) 

    # reduced model should have at leat 50% match compared to the original
    # For the test, we reduced the training minibatches, thus the match is lower.
    assert(original_prediction_percentage * 0.5 <= _percentage_match(labels, predicted_label_probs))
