# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os, sys
import numpy as np
from cntk import load_model
from cntk.device import try_set_default_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "SequenceToSequence", "CMUDict", "Python"))

TOLERANCE_ABSOLUTE = 1E-2 # after 1 epoch with non-reproducable random init, we need a large tolerance (per recommendation by mhilleb)

def test_sequence_to_sequence(device_id):

    # import code after setting the device, otherwise some part of the code picks up "default device"
    # which causes an inconsistency if there is already another job using GPU #0
    from Sequence2Sequence import create_reader, DATA_DIR, MODEL_DIR, TRAINING_DATA, VALIDATION_DATA, TESTING_DATA, \
                                  VOCAB_FILE, get_vocab, create_model, model_path_stem, train, evaluate_metric
    from cntk.ops.tests.ops_test_utils import cntk_device
    try_set_default_device(cntk_device(device_id))

    # hook up data (train_reader gets False randomization to get consistent error)
    train_reader = create_reader(os.path.join(DATA_DIR, TRAINING_DATA), False)
    valid_reader = create_reader(os.path.join(DATA_DIR, VALIDATION_DATA), True)
    test_reader  = create_reader(os.path.join(DATA_DIR, TESTING_DATA), False)
    vocab, i2w, _ = get_vocab(os.path.join(DATA_DIR, VOCAB_FILE))

    # create model
    model = create_model()

    # train (with small numbers to finish within a reasonable amount of time)
    train(train_reader, valid_reader, vocab, i2w, model, max_epochs=1, epoch_size=5000)

    # now test the model and print out test error (for automated test)
    model_filename = os.path.join(MODEL_DIR, model_path_stem + ".cmf.0")
    model = load_model(model_filename)
    error = evaluate_metric(test_reader, model, 10)

    print(error)

    #expected_error =  0.9943119920022192 # when run separately
    expected_error =  0.9912881900980582 # when run inside the harness--random-initialization?
    assert np.allclose(error, expected_error, atol=TOLERANCE_ABSOLUTE)
