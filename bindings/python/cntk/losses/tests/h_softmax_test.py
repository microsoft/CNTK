# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

"""
Unit tests for the hierarchical softmax class.
"""

import numpy as np
import cntk as C
from _cntk_py import set_fixed_random_seed

def test_h_softmax():
  set_fixed_random_seed(1)
  input_dim = 2
  num_output_classes = 4
  minibatch_size = 3

  # neural network structure for hierarchical softmax
  h_input = C.input_variable(input_dim)
  h_target = C.input_variable([1])
  h_z, class_probs, all_probs = C.hierarchical_softmax_layer(h_input, h_target, num_output_classes)

  a = np.reshape(np.arange(minibatch_size * input_dim, dtype = np.float32), (minibatch_size, input_dim))
  labels = np.reshape(np.arange(minibatch_size, dtype = np.float32), (minibatch_size, 1)) % num_output_classes
  val_z = h_z.eval({h_input: a, h_target: labels})
  val_class_probs = class_probs.eval({h_input: a, h_target: labels})
  val_all_probs = [x.eval({h_input: a, h_target: labels}) for x in all_probs]

  expected_z = [[[0.17082828]], [[0.17143427]], [[0.0001837]]]
  expected_class_probs = [[ 0.4046618 ,  0.59533817],
                        [ 0.23773022,  0.76226979],
                        [ 0.12518175,  0.87481827]]

  expected_all_probs =  [[[ 0.17082828,  0.23383351],
                          [ 0.06629595,  0.17143427],
                          [ 0.02127092,  0.10391083]],
                        [[1.76951319e-01, 4.18386817e-01],
                          [7.11729145e-03, 7.55152524e-01],
                          [1.83700817e-04, 8.74634564e-01]]]

  assert np.allclose(expected_z, val_z)
  assert np.allclose(expected_class_probs, val_class_probs)
  assert np.allclose(expected_all_probs, val_all_probs)
