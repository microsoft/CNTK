# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import subprocess
import sys
import numpy as np
import cntk as C

def mpiexec_execute(script, mpiexec_params):
    timeout_seconds = 300
    cmd = ['mpiexec'] + mpiexec_params + ['python', script]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    if sys.version_info[0] < 3:
        out = p.communicate()[0]
    else:
        try:
            out = p.communicate(timeout=timeout_seconds)[0]  # in case we have a hang
        except subprocess.TimeoutExpired:
            os.kill(p.pid, signal.CTRL_C_EVENT)
            raise RuntimeError('Timeout in mpiexec, possibly hang')
    return out.decode(sys.getdefaultencoding())

def test_distributed_eval():
    try:
        str_out = mpiexec_execute(__file__, ["-n", "2"])
    except AssertionError:
        print(str_out)
        raise

if __name__=='__main__':
    try:
        input_dim = 2
        proj_dim = 2

        x = C.input_variable(shape=(input_dim,))
        W = C.parameter(shape=(input_dim, proj_dim), init=[[1, 0], [0, 1]])
        B = C.parameter(shape=(proj_dim,), init=[[0, 1]])
        t = C.times(x, W)
        z = t + B

        labels = C.input_variable(shape=(proj_dim,))
        pe = C.metrics.classification_error(z, labels)

        tester = C.eval.Evaluator(pe)

        if C.distributed.Communicator.rank() == 1:
            x_value = [[0, 1], [2, 2]]
            label_value = [[0, 1], [1, 0]]
        else:
            x_value = [[1, 2], [2, 3]]
            label_value = [[1, 0], [1, 0]]

        arguments = {
            x: np.asarray(x_value, dtype = np.float32),
            labels: np.asarray(label_value, dtype = np.float32)}
        eval_error = tester.test_minibatch(arguments, distributed=True)
        expected_eval_error = 0.75
        assert np.allclose(eval_error, expected_eval_error), \
            "Evaluation error is %f, expected %f." % (eval_error, expected_eval_error)
    finally:
        C.distributed.Communicator.finalize()
