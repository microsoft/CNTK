import sys
import os
import numpy as np
from ReasoNet.model import text_convolution 
from cntk.blocks import Placeholder, Constant
from cntk.ops import input_variable

def testTextConvolution():
  text = np.reshape(np.arange(25.0, dtype=np.float32), (1, 5,5))
  x = input_variable(shape=(1, 5, 5,)) 
  c = text_convolution(3, 5, 5)(x)
  v = c.eval([text])[0]
  print(v)
