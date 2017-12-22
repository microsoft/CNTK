# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os
import sys
from cntk import Function
from cntk import parameter, plus
from cntk.io import MinibatchSource, StreamDef, StreamDefs
from cntk.losses import squared_error
import re
import cntk as C

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)

from distributed_common import mpiexec_execute

def test_sample_count_with_several_distributed_learners():
    str_out = mpiexec_execute(__file__, ["-n", "2"], [])

    results = re.findall("Completed with exception.", str_out)
    if len(results) != 0:
        print(str_out)
        assert False

    results = re.findall("Completed successfully.", str_out)
    if len(results) != 2:
        print(str_out)
        assert False

if __name__=='__main__':
    in1 = C.input_variable(shape=1)
    labels = C.input_variable(shape=1)
    p1 = parameter(shape=1)
    p2 = parameter(shape=1)
    n = plus(in1, p1, name='n')
    z = plus(n, p2, name='z')
    ce = squared_error(z, labels)

    momentum_schedule = C.momentum_schedule_per_sample(0.9990913221888589)
    lr_per_sample = C.learning_parameter_schedule_per_sample(0.007)
    dist_learners = [
        C.distributed.data_parallel_distributed_learner(C.momentum_sgd([p1], lr_per_sample, momentum_schedule, True)),
        C.distributed.data_parallel_distributed_learner(C.momentum_sgd([p2], lr_per_sample, momentum_schedule, True))
    ]

    trainer = C.Trainer(z, ce, dist_learners)
    in1_value = [[1]]
    label_value = [[0]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output

    def check_samples(learners, expected_number_of_samples):
        for learner in learners:
            if learner.total_number_of_samples_seen != expected_number_of_samples:
                print("Completed with exception.")
                raise ValueError("%d samples expected, got %d" % (expected_number_of_samples, learner.total_number_of_samples_seen))

    trainer.train_minibatch(arguments, outputs=[z_output])
    check_samples(dist_learners, 2)

    trainer.train_minibatch(arguments, outputs=[z_output])
    check_samples(dist_learners, 4)

    print("Completed successfully.")
    C.distributed.Communicator.finalize()    
