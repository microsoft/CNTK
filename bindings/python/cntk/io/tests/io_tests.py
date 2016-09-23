
# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import numpy as np
from cntk.io import text_format_minibatch_source, StreamConfiguration

abs_path = os.path.dirname(os.path.abspath(__file__))

def test_text_format():

    # 0	|x 560:1	|y 1 0 0 0 0
    # 0	|x 0:1
    # 0	|x 0:1
    # 1	|x 560:1	|y 0 1 0 0 0
    # 1	|x 0:1
    # 1	|x 0:1
    # 1	|x 424:1
    path = os.path.join(abs_path, 'tf_data.txt')

    input_dim = 1000
    num_output_classes = 5

    mb_source = text_format_minibatch_source(path, [
                    StreamConfiguration( 'features', input_dim, True, 'x' ),
                    StreamConfiguration( 'labels', num_output_classes, False, 'y')], 0)

    features_si = mb_source.stream_info('features')
    labels_si = mb_source.stream_info('labels')

    mb = mb_source.get_next_minibatch(7)

    features = mb[features_si].m_data
    # 2 samples, max seq len 4, 1000 dim
    assert features.data().shape().dimensions() == (2, 4, input_dim)
    assert features.data().is_sparse()
    # TODO features is sparse and cannot be accessed right now:
    # *** RuntimeError: DataBuffer/WritableDataBuffer methods can only be called for NDArrayiew objects with dense storage format

    labels = mb[labels_si].m_data
    # 2 samples, max seq len 1, 5 dim
    assert labels.data().shape().dimensions() == (2, 1, num_output_classes)
    assert not labels.data().is_sparse()

    assert np.allclose(labels.data().to_numpy(), 
            np.asarray([
                [[ 1.,  0.,  0.,  0.,  0.]],
                [[ 0.,  1.,  0.,  0.,  0.]]
                ]))

