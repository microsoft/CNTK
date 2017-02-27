# ==============================================================================
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from cntk.ops import *
from cntk.ops.functions import load_model

def create_model(tmpdir):
    i1 = input_variable((1,2), name='i1')
    root_node = abs(i1)

    filename = str(tmpdir + '/load_and_eval.dnn')
    root_node.save_model(filename)
    print('Save the model to ' + filename)
    return filename

def load_and_eval_model(filename):
    model = load_model(filename)
    input1 = [[[-10,20]]]
    i1 = model.arguments[0]
    result = model.eval({i1: input1})
    print('Evaluation result:')
    print(result)
    expected = [[[[10,20]]]]
    assert np.allclose(result, expected)

if __name__=='__main__':
    #TODO: use a better tmp directory 
    modelfile = create_model('.')
    load_and_eval_model(modelfile)
