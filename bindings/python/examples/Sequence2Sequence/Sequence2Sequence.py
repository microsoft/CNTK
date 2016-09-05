# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import sys
import os
import time
from cntk import learning_rates_per_sample, DeviceDescriptor, Trainer, sgdlearner, Axis, get_train_loss, get_train_eval_criterion
from cntk.ops import variable, cross_entropy_with_softmax, classification_error
from examples.common.nn import LSTMP_component_with_self_stabilization, embedding, fully_connected_linear_layer, select_last

# Creates and trains a sequence to sequence translation model
def train_sequence_to_sequence_translator():

    input_vocab_dim = 69
    label_vocab_dim = 69

    hidden_dim = 512
    num_layers = 2

    # Source and target inputs to the model
    input_dynamic_axes = [ Axis('inputAxis'), Axis.default_batch_axis() ]
    raw_input = input_variable(shape=(input_vocab_dim), dynamic_axes = input_dynamic_axes)

    label_dynamic_axes = [ Axis('labelAxis'), Axis.default_batch_axis() ]
    raw_labels = input_variable(shape=(label_vocab_dim), dynamic_axes = label_dynamic_axes)

    input_sequence = raw_input

    # Drop the sentence start token from the label, for decoder training
    label_sequence = cntk.ops.slice(raw_labels, label_dynamic_axes[0], 1, 0)
    label_sentence_start = Sequence.first(raw_labels)

    is_first_label = Sequence.is_first(label_sequence)

    label_sentence_start_scattered = Sequence.scatter(label_sentence_start, is_first_label)

    # Encoder
    encoderOutputH = stabilize<float>(inputEmbedding, device)
    futureValueRecurrenceHook = [](const Variable& x) { return FutureValue(x) }
    for (size_t i = 0 i < num_layers ++i)
        std::tie(encoderOutputH, encoderOutputC) = LSTMPComponentWithSelfStabilization<float>(encoderOutputH, hidden_dim, hidden_dim, futureValueRecurrenceHook, futureValueRecurrenceHook, device)

    thoughtVectorH = Sequence::First(encoderOutputH)
    thoughtVectorC = Sequence::First(encoderOutputC)

    thoughtVectorBroadcastH = Sequence::BroadcastAs(thoughtVectorH, labelEmbedding)
    thoughtVectorBroadcastC = Sequence::BroadcastAs(thoughtVectorC, labelEmbedding)

    /* Decoder */
    bool addBeamSearchReorderingHook = false
    beamSearchReorderHook = Constant({ 1, 1 }, 1.0f)
    decoderHistoryFromGroundTruth = labelEmbedding
    decoderInput = ElementSelect(is_first_label, label_sentence_startEmbeddedScattered, PastValue(decoderHistoryFromGroundTruth))

    decoderOutputH = Stabilize<float>(decoderInput, device)
    FunctionPtr decoderOutputC
    pastValueRecurrenceHookWithBeamSearchReordering = [addBeamSearchReorderingHook, beamSearchReorderHook](const FunctionPtr& operand) {
        return PastValue(addBeamSearchReorderingHook ? Times(operand, beamSearchReorderHook) : operand)
    }

    for (size_t i = 0 i < num_layers ++i)
    {
        std::function<FunctionPtr(const Variable&)> recurrenceHookH, recurrenceHookC
        if (i == 0)
        {
            recurrenceHookH = pastValueRecurrenceHookWithBeamSearchReordering
            recurrenceHookC = pastValueRecurrenceHookWithBeamSearchReordering
        }
        else
        {
            isFirst = Sequence::IsFirst(labelEmbedding)
            recurrenceHookH = [labelEmbedding, thoughtVectorBroadcastH, isFirst, addBeamSearchReorderingHook, beamSearchReorderHook](const FunctionPtr& operand) {
                return ElementSelect(isFirst, thoughtVectorBroadcastH, PastValue(addBeamSearchReorderingHook ? Times(operand, beamSearchReorderHook) : operand))
            }

            recurrenceHookC = [labelEmbedding, thoughtVectorBroadcastC, isFirst, addBeamSearchReorderingHook, beamSearchReorderHook](const FunctionPtr& operand) {
                return ElementSelect(isFirst, thoughtVectorBroadcastC, PastValue(addBeamSearchReorderingHook ? Times(operand, beamSearchReorderHook) : operand))
            }
        }

        std::tie(decoderOutputH, encoderOutputC) = LSTMPComponentWithSelfStabilization<float>(decoderOutputH, hidden_dim, hidden_dim, recurrenceHookH, recurrenceHookC, device)
    }

    decoderOutput = decoderOutputH
    decoderDim = hidden_dim

    /* Softmax output layer */
    outputLayerProjWeights = Parameter(NDArrayView::RandomUniform<float>({ label_vocab_dim, decoderDim }, -0.05, 0.05, 1, device))
    biasWeights = Parameter({ label_vocab_dim }, 0.0f, device)

    z = Plus(Times(outputLayerProjWeights, Stabilize<float>(decoderOutput, device)), biasWeights, L"classifierOutput")
    ce = CrossEntropyWithSoftmax(z, label_sequence, L"lossFunction")
    errs = ClassificationError(z, label_sequence, L"classificationError")




    input_dim = 2000
    cell_dim = 25
    hidden_dim = 25
    embedding_dim = 50
    num_output_classes = 5

    # Input variables denoting the features and label data
    features = variable(shape=input_dim, is_sparse=True, name="features")
    label = variable(num_output_classes, dynamic_axes = [Axis.default_batch_axis()], name="labels")

    # Instantiate the sequence classification model
    classifier_output = LSTM_sequence_classifer_net(features, num_output_classes, embedding_dim, hidden_dim, cell_dim)

    ce = cross_entropy_with_softmax(classifier_output, label)
    pe = classification_error(classifier_output, label)
    
    rel_path = r"../../../../Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)

    mb_source = text_minibatch_source(path, [ ( 'features', input_dim, True, 'x' ), ( 'labels', num_output_classes, False, 'y' ) ], 0)
    features_si = mb_source.stream_info(features)
    labels_si = mb_source.stream_info(label)

    # Instantiate the trainer object to drive the model training
    lr = lr = learning_rates_per_sample(0.0005)
    trainer = Trainer(classifier_output, ce, pe, [sgdlearner(classifier_output.owner.parameters(), lr)])                   

    # Get minibatches of sequences to train with and perform model training
    minibatch_size = 200
    training_progress_output_freq = 1  
    i = 0
    while True:
        mb = mb_source.get_next_minibatch(minibatch_size)
        if  len(mb) == 0:
            break

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        arguments = {features : mb[features_si].m_data, label : mb[labels_si].m_data}
        trainer.train_minibatch(arguments)

        print_training_progress(training_progress_output_freq, i, trainer)

        i += 1

if __name__=='__main__':    
    # Specify the target device to be used for computing
    target_device = DeviceDescriptor.cpu_device()
    DeviceDescriptor.set_default_device(target_device)

    train_sequence_classifier()
