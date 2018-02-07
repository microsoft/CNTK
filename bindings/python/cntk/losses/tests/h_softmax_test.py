# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the hierarchical softmax class.
"""

import numpy as np
import cntk as C
from math import sqrt, ceil

def test_h_softmax():
  input_dim = 2
  num_output_classes = 4
  minibatch_size = 3
  n_classes = int(ceil(sqrt(num_output_classes)))
  n_outputs_per_class = n_classes

  w1 = C.parameter(shape=(input_dim, n_classes), init=C.glorot_normal(seed=1), name='w1')
  b1 = C.parameter(shape=(n_classes), init=C.glorot_normal(seed=2), name='b1')
  w2s = C.parameter(shape=(n_classes, input_dim, n_outputs_per_class), init=C.glorot_normal(seed=3), name='w2s')
  b2s = C.parameter(shape=(n_classes, n_outputs_per_class), init=C.glorot_normal(seed=4), name='b2s')

  # neural network structure for hierarchical softmax
  h_input = C.input_variable(input_dim)
  h_target_class = C.input_variable([1])
  h_target_output_in_class = C.input_variable([1])
  h_z, class_probs, all_probs = C.hierarchical_softmax_layer(h_input, num_output_classes, h_target_class, h_target_output_in_class, minibatch_size, w1, b1, w2s, b2s)

  a = np.reshape(np.arange(minibatch_size * input_dim, dtype = np.float32), (minibatch_size, input_dim))
  labels = np.reshape(np.arange(minibatch_size, dtype = np.float32), (minibatch_size, 1)) % num_output_classes
  target_labels = labels // n_outputs_per_class
  target_output_in_labels = labels % n_outputs_per_class
  val_z = h_z.eval({h_input: a, h_target_class: target_labels, h_target_output_in_class: target_output_in_labels})
  val_class_probs = class_probs.eval({h_input: a, h_target_class: target_labels, h_target_output_in_class: target_output_in_labels})
  val_all_probs = [x.eval({h_input: a, h_target_class: target_labels, h_target_output_in_class: target_output_in_labels}) for x in all_probs]

  expected_z = [[0.0313047], [0.00323934], [0.99006385]]
  expected_class_probs = [[ 0.04346574,  0.95653421],
                          [ 0.0204236 ,  0.97957635],
                          [ 0.0094756 ,  0.99052447]]
  expected_all_probs =  [[[ 0.0313047 ,  0.01216104],
                          [ 0.01718426,  0.00323934],
                          [ 0.00868148,  0.00079412]],
                          [[  5.82283854e-01,   3.74250382e-01],
                          [  9.62925494e-01,   1.66507624e-02],
                          [  9.90063846e-01,   4.60594223e-04]]]                       

  assert np.allclose(expected_z, val_z)
  assert np.allclose(expected_class_probs, val_class_probs)
  assert np.allclose(expected_all_probs, val_all_probs)

def test_h_softmax_for_sequence():
  input_dim = 2
  num_output_classes = 4
  minibatch_size = 3
  seq_size = 2
  n_classes = int(ceil(sqrt(num_output_classes)))
  n_outputs_per_class = n_classes

  w1 = C.parameter(shape=(input_dim, n_classes), init=C.glorot_normal(seed=2), name='w1')
  b1 = C.parameter(shape=(n_classes), init=C.glorot_normal(seed=3), name='b1')
  w2s = C.parameter(shape=(n_classes, input_dim, n_outputs_per_class), init=C.glorot_normal(seed=4), name='w2s')
  b2s = C.parameter(shape=(n_classes, n_outputs_per_class), init=C.glorot_normal(seed=5), name='b2s')

  # neural network structure for hierarchical softmax
  h_input = C.sequence.input_variable(input_dim)
  h_target_class = C.sequence.input_variable([1])
  h_target_output_in_class = C.sequence.input_variable([1])
  h_z, class_probs, all_probs = C.hierarchical_softmax_layer_for_sequence(h_input, num_output_classes, h_target_class, h_target_output_in_class, minibatch_size, w1, b1, w2s, b2s)

  a = np.reshape(np.arange(seq_size * minibatch_size * input_dim, dtype = np.float32), (seq_size, minibatch_size, input_dim))
  labels = np.reshape(np.arange(seq_size * minibatch_size, dtype = np.float32), (seq_size, minibatch_size, 1)) % num_output_classes
  target_labels = labels // n_outputs_per_class
  target_output_in_labels = labels % n_outputs_per_class
  val_z = h_z.eval({h_input: a, h_target_class: target_labels, h_target_output_in_class: target_output_in_labels})
  val_class_probs = class_probs.eval({h_input: a, h_target_class: target_labels, h_target_output_in_class: target_output_in_labels})
  val_all_probs = [x.eval({h_input: a, h_target_class: target_labels, h_target_output_in_class: target_output_in_labels}) for x in all_probs]

  expected_z = [[[ 0.16448107],
            [ 0.00597861],
            [ 0.99322051]],
            [[  8.59128195e-04],
            [  3.77086673e-09],
            [  3.42400197e-12]]]
  expected_class_probs = [[[  5.81252098e-01,   4.18747932e-01],
                          [  1.03938626e-02,   9.89606142e-01],
                          [  7.94661901e-05,   9.99920487e-01]],
                          [[  6.01340048e-07,   9.99999404e-01],
                          [  4.55011762e-09,   1.00000000e+00],
                          [  3.44291574e-11,   1.00000000e+00]]]
  expected_all_probs =  [[[[  1.64481074e-01,   4.16771024e-01],
                          [  4.41524992e-03,   5.97861316e-03],
                          [  4.61043091e-05,   3.33618809e-05]],
                          [[  4.33648694e-07,   1.67691354e-07],
                          [  3.77086673e-09,   7.79251219e-10],
                          [  3.10051568e-11,   3.42400197e-12]]],
                          [[[ 0.29590073,  0.12284722],
                          [ 0.93986785,  0.04973821],
                          [ 0.99322051,  0.00669997]],
                          [[  9.99140263e-01,   8.59128195e-04],
                          [  9.99890447e-01,   1.09594235e-04],
                          [  9.99986053e-01,   1.39711719e-05]]]]
                     
  assert np.allclose(expected_z, val_z)
  assert np.allclose(expected_class_probs, val_class_probs)
  assert np.allclose(expected_all_probs, val_all_probs)
