import os
import cntk as C
import numpy as np


# Select the right target device
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))

C.device.try_set_default_device(C.device.gpu(0))

data_dir = os.path.join("..", "Tests", "EndToEndTests", "Speech", "Data")
print("Current directory {0}".format(os.getcwd()))
if os.path.realpath(data_dir) != os.path.realpath(os.getcwd()):
    print("Changing to data directory {0}".format(data_dir))
    os.chdir(data_dir)

feature_dimension = 33
feature = C.sequence.input((feature_dimension))

label_dimension = 133
label = C.sequence.input((label_dimension))

train_feature_filepath = "glob_0000.scp"
train_label_filepath = "glob_0000.mlf"
mapping_filepath = "state_ctc.list"
train_feature_stream = C.io.HTKFeatureDeserializer(C.io.StreamDefs(amazing_feature = C.io.StreamDef(shape = feature_dimension, scp = train_feature_filepath)))
train_label_stream = C.io.HTKMLFDeserializer(mapping_filepath, C.io.StreamDefs(awesome_label = C.io.StreamDef(shape = label_dimension, mlf = train_label_filepath)))
train_data_reader = C.io.MinibatchSource([train_feature_stream, train_label_stream], frame_mode = False)
train_input_map = {feature: train_data_reader.streams.amazing_feature, label: train_data_reader.streams.awesome_label}

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
max_epochs = 50

#import pdb;pdb.set_trace()

criteria = C.forward_backward(C.labels_to_graph(label), z, blankTokenId=132, delayConstraint=3)
ce = C.cross_entropy_with_softmax(z, label)
err = C.classification_error(z, label)
lr = C.learning_rate_schedule(.01, C.UnitType.minibatch)
mm = C.momentum_schedule([(10, 0.9), (0, 0.99)], mbsize)
learner = C.momentum_sgd(z.parameters, lr, mm)
trainer = C.Trainer(z, (criteria, err), learner)

C.logging.log_number_of_parameters(z)
progress_printer = C.logging.progress_print.ProgressPrinter(tag='Training', num_epochs = max_epochs)

for epoch in range(max_epochs):
	for mb in range(mbs_per_epoch):
		minibatch = train_data_reader.next_minibatch(mbsize, input_map = train_input_map)
		trainer.train_minibatch(minibatch)
		progress_printer.update_with_trainer(trainer, with_metric = True)

	print('Trained on a total of ' + str(trainer.total_number_of_samples_seen) + ' frames')
	progress_printer.epoch_summary(with_metric = True)

z.save('CTC_' + str(max_epochs) + 'epochs_' + str(mbsize) + 'mbsize_' + str(mbs_per_epoch) + 'mbs.model')