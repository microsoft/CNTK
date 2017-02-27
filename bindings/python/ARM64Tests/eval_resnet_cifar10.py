# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import os
import argparse
import math
import numpy as np

from cntk.ops import softmax
from cntk.ops.functions import load_model
from PIL import Image 
import numpy as np

z = load_model("resnet20.dnn")
image_path = "zebra.jpg"
img = Image.open(image_path)
resized = img.resize((32, 32), Image.ANTIALIAS)
rgb_image = np.asarray(resized, dtype=np.float32) - 128
bgr_image = rgb_image[..., [2, 1, 0]]
pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

output = z.eval({z.arguments[0]: [pic]})
print("Eval ouput:")
print(np.squeeze(output))
sm = softmax(output[0, 0])
print("\nSoftmax output:")
print(sm.eval())

