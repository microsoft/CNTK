# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import cntk as C
import numpy as np
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device
import pytest

TOLERANCE_ABSOLUTE = 1E-2

def test_lattice_deserializer(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU')
    try_set_default_device(cntk_device(device_id))

    data_dir = r"D:\users\vadimma\cntk_tut_deprecated\CNTK\Tests\EndToEndTests\Speech\Data\AN4Corpus\v0"

    feature_dimension = 33
    feature = C.sequence.input_variable(feature_dimension)

    label_dimension = 133
    label = C.sequence.input_variable(label_dimension)

    axis_lattice = C.Axis.new_unique_dynamic_axis('lattice_axis')
    lattice = C.sequence.input_variable(1, sequence_axis=axis_lattice)

    train_feature_filepath = os.path.join(data_dir,"glob_0000.scp")
    train_label_filepath = os.path.join(data_dir,"glob_0000.mlf")
    train_lattice_index_path = os.path.join(data_dir,"latticeIndex.txt")
    mapping_filepath = os.path.join(data_dir,"state.list")
    train_feature_stream = C.io.HTKFeatureDeserializer(
    C.io.StreamDefs(speech_feature = C.io.StreamDef(shape = feature_dimension, scp = train_feature_filepath)))
    train_label_stream = C.io.HTKMLFDeserializer(
    mapping_filepath, C.io.StreamDefs(speech_label = C.io.StreamDef(shape = label_dimension, mlf = train_label_filepath)), True)
    train_lattice_stream = C.io.LatticeDeserializer(train_lattice_index_path,C.io.StreamDefs(speech_lattice = C.io.StreamDef()))
    train_data_reader = C.io.MinibatchSource([train_feature_stream, train_label_stream, train_lattice_stream], frame_mode = False)
    train_input_map = {feature: train_data_reader.streams.speech_feature, label: train_data_reader.streams.speech_label, lattice: train_data_reader.streams.speech_lattice}

    feature_mean = np.fromfile(os.path.join("GlobalStats", "mean.363"), dtype=float, count=feature_dimension)
    feature_inverse_stddev = np.fromfile(os.path.join("GlobalStats", "var.363"), dtype=float, count=feature_dimension)

    feature_normalized = (feature - feature_mean) * feature_inverse_stddev

    with C.default_options(activation=C.sigmoid):
        z = C.layers.Sequential([
            C.layers.For(range(3), lambda: C.layers.Recurrence(C.layers.LSTM(1024))),
            C.layers.Dense(label_dimension)
        ])(feature_normalized)
    mbsize = 1024
    mbs_per_epoch = 10
    max_epochs = 2

    symListPath = os.path.join(data_dir,"CY2SCH010061231_1369712653.numden.lats.symlist")
    phonePath = os.path.join(data_dir,"model.overalltying")
    stateListPath = os.path.join(data_dir,"state.list")
    transProbPath = os.path.join(data_dir,"model.transprob")

    criteria = C.lattice_sequence_with_softmax(label, z, z, lattice, symListPath, phonePath, stateListPath, transProbPath)
    err = C.classification_error(label,z)
    # Learning rate parameter schedule per sample:
    # Use 0.01 for the first 3 epochs, followed by 0.001 for the remaining
    lr = C.learning_parameter_schedule_per_sample([(3, .01), (1,.001)])
    mm = C.momentum_schedule([(1000, 0.9), (0, 0.99)], mbsize)
    learner = C.momentum_sgd(z.parameters, lr, mm)
    trainer = C.Trainer(z, (criteria, err), learner)

    C.logging.log_number_of_parameters(z)
    progress_printer = C.logging.progress_print.ProgressPrinter(tag='Training', num_epochs = max_epochs)


    for epoch in range(max_epochs):
        for mb in range(mbs_per_epoch):
    #        import pdb;pdb.set_trace()
            minibatch = train_data_reader.next_minibatch(mbsize, input_map = train_input_map)
            trainer.train_minibatch(minibatch)
            progress_printer.update_with_trainer(trainer, with_metric = True)

        print('Trained on a total of ' + str(trainer.total_number_of_samples_seen) + ' frames')
        progress_printer.epoch_summary(with_metric = True)

    assert np.allclose(trainer.previous_minibatch_evaluation_average, 0.15064, atol=TOLERANCE_ABSOLUTE)
    assert np.allclose(trainer.previous_minibatch_loss_average, 0.035923, atol=TOLERANCE_ABSOLUTE)
    assert (trainer.previous_minibatch_sample_count == 218)
    assert (trainer.total_number_of_samples_seen == 5750)
    print("Completed successfully.")

if __name__=='__main__':
    test_lattice_deserializer(0)