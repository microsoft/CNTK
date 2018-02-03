#!/usr/bin/python3

import sys
import numpy as np

(_, ref, indir, out) = sys.argv

print('converting ref model', ref, '<-', indir, '>', out, '...')

# reference model. We use names and shapes, but discard all actual values
data = np.load(ref)

# convert
new_model = dict()
for key in data.keys():
  if key == "special:model.yml":
    print('retaining object named', key)
    new_model[key] = data[key]
    continue
  param = data[key]
  file = indir + '/' + key
  file_data  = np.fromfile(file, dtype=np.float32, count=-1) # -1 means use file length; allows us to validate
  new_param = file_data.reshape(param.shape)
  new_model[key] = new_param

# save
np.savez(out, **new_model)
print(len(new_model.keys()), 'parameter tensors written to', out)
