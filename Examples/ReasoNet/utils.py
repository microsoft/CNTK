import numpy as np

class uniform_initializer:
  def __init__(self, scale=1, bias=0, seed=0):
    self.seed = seed
    self.scale = scale
    self.bias = bias
    np.random.seed(self.seed)

  def reset(self):
    np.random.seed(self.seed)

  def next(self, size=None):
    return np.random.uniform(0, 1, size)*self.scale + self.bias
