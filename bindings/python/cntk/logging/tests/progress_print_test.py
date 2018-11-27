# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import cntk as C

def test_tensorboard_write_image(tmpdir):
    image_height = 28
    image_width = 28
    num_channels = 1
    input_var = C.ops.input_variable((num_channels, image_height, image_width), np.float32)
    input_batch = np.zeros((2, num_channels, image_height, image_width), dtype=np.float32)

    # Write image to TensorBoardProgressWriter
    dict = {input_var : input_batch}
    log_dir = str(tmpdir / 'log')
    tensorboard_writer = C.logging.TensorBoardProgressWriter(log_dir=log_dir)
    tensorboard_writer.write_image('test', dict, 0)
    tensorboard_writer.flush();
    tensorboard_writer.close();
