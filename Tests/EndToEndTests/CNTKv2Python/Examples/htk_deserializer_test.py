import cntk as C
from cntk.io import MinibatchSource, HTKFeatureDeserializer, HTKMLFDeserializer, HTKMLFBinaryDeserializer, StreamDef, StreamDefs
from cntk.layers import Recurrence, Dense, LSTM, Sequential, For

import os
abs_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Speech", "AN4", "Data")
e2e_data_path="../../../../Tests/EndToEndTests/Speech/Data/"

def test_htk_deserializers():
    mbsize = 640
    epoch_size = 1000 * mbsize
    lr = [0.001]

    feature_dim = 33
    num_classes = 132
    context = 2

    os.chdir(data_path)

    features_file = "glob_0000.scp"
    labels_file = "glob_0000.mlf"
    label_mapping_file = "state.list"

    fd = HTKFeatureDeserializer(StreamDefs(
        amazing_features = StreamDef(shape=feature_dim, context=(context,context), scp=features_file)))

    ld = HTKMLFDeserializer(label_mapping_file, StreamDefs(
        awesome_labels = StreamDef(shape=num_classes, mlf=labels_file)))

    reader = MinibatchSource([fd,ld])

    features = C.sequence.input_variable(((2*context+1)*feature_dim))
    labels = C.sequence.input_variable((num_classes))

    model = Sequential([For(range(3), lambda : Recurrence(LSTM(256))),
                        Dense(num_classes)])
    z = model(features)
    ce = C.cross_entropy_with_softmax(z, labels)
    errs = C.classification_error    (z, labels)

    learner = C.fsadagrad(z.parameters,
                          lr=C.learning_parameter_schedule_per_sample(lr, epoch_size=epoch_size),
                          momentum=C.momentum_schedule_per_sample(0.9990913221888589),
                          gradient_clipping_threshold_per_sample=15, gradient_clipping_with_truncation=True)
    progress_printer = C.logging.ProgressPrinter(freq=0)
    trainer = C.Trainer(z, (ce, errs), learner, progress_printer)

    input_map={ features: reader.streams.amazing_features, labels: reader.streams.awesome_labels }

    # just run and verify it doesn't crash
    for i in range(3):
        mb_data = reader.next_minibatch(mbsize, input_map=input_map)
        trainer.train_minibatch(mb_data)
    assert True
    os.chdir(abs_path)

def test_multiple_mlf_files():
    os.chdir(data_path)

    feature_dim = 33
    num_classes = 132
    context = 2

    test_mlf_path = e2e_data_path+"glob_00001.mlf"

    features_file = "glob_0000.scp"
    label_files = [ "glob_0000.mlf", test_mlf_path]
    label_mapping_file = "state.list"

    fd = HTKFeatureDeserializer(StreamDefs(
        amazing_features = StreamDef(shape=feature_dim, context=(context,context), scp=features_file)))

    ld = HTKMLFDeserializer(label_mapping_file, StreamDefs(
        awesome_labels = StreamDef(shape=num_classes, mlf=label_files)))

    # Make sure we can read at least one minibatch.
    mbsource = MinibatchSource([fd,ld])
    mbsource.next_minibatch(1)

    os.chdir(abs_path)

def test_mlf_binary_files():
    os.chdir(data_path)

    feature_dim = 33
    num_classes = 132
    context = 2

    features_file = "glob_0000.scp"

    fd = HTKFeatureDeserializer(StreamDefs(
        amazing_features = StreamDef(shape=feature_dim, context=(context,context), scp=features_file)))

    ld = HTKMLFBinaryDeserializer(StreamDefs(awesome_labels = StreamDef(shape=num_classes, mlf=e2e_data_path + "mlf2.bin")))

    # Make sure we can read at least one minibatch.
    mbsource = MinibatchSource([fd,ld])
    mbsource.next_minibatch(1)

    os.chdir(abs_path)

def test_multiple_mlf_binary_files():
    os.chdir(data_path)

    feature_dim = 33
    num_classes = 132
    context = 2

    features_file = "glob_0000.scp"
    label_files = [ e2e_data_path+"mlf2.bin", e2e_data_path+"mlf2.bin"]

    fd = HTKFeatureDeserializer(StreamDefs(
        amazing_features = StreamDef(shape=feature_dim, context=(context,context), scp=features_file)))

    ld = HTKMLFBinaryDeserializer(StreamDefs(awesome_labels = StreamDef(shape=num_classes, mlf=label_files)))

    # Make sure we can read at least one minibatch.
    mbsource = MinibatchSource([fd,ld])
    mbsource.next_minibatch(1)

    os.chdir(abs_path)

def test_multiple_streams_in_htk():
    feature_dim = 33
    context = 2

    os.chdir(data_path)

    features_file = "glob_0000.scp"

    fd = HTKFeatureDeserializer(StreamDefs(
        amazing_features = StreamDef(shape=feature_dim, context=(context,context), scp=features_file),
        amazing_features2 = StreamDef(shape=feature_dim, context=(context,context), scp=features_file)))

    mbs = MinibatchSource([fd])
    mb = mbs.next_minibatch(1)
    assert (mb[mbs.streams.amazing_features].asarray() == mb[mbs.streams.amazing_features2].asarray()).all()
    os.chdir(abs_path)
