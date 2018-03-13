# CNTK Current Iteration

## Change profiler details output format to be chrome://tracing

## Enable per-node timing. Working example [here](../Examples/Image/Classification/MLP/Python/SimpleMNIST.py)
- per-node timing creates items in profiler details when profiler is enabled.
- usage in Python:
```
import cntk as C
C.debugging.debug.set_node_timing(True)
C.debugging.start_profiler() # optional
C.debugging.enable_profiler() # optional
#<trainer|evaluator|function> executions
<trainer|evaluator|function>.print_node_timing()
C.debugging.stop_profiler()
```

## CPU inference performance improvements using MKL
- Accelerates some common tensor ops in Intel CPU inference for float32, especially for fully connected networks
- Can be turned on/off by cntk.cntk_py.enable_cpueval_optimization()/cntk.cntk_py.disable_cpueval_optimization()

## 1BitSGD incorporated into CNTK
- 1BitSGD source code is now available with CNTK license (MIT license) under Source/1BitSGD/
- 1bitsgd build target was merged into existing gpu target

## New loss function: hierarchical softmax (Thanks @yaochengji for the contribution!)

## Distributed Training with Mulitple Learners
- Trainer now accepts multiple parameter learners for distributed training. With this change, different parameters of a network can be learned by different learners in a single training session. This also facilitates distributed training for GANs. For more information, please refer to the [Basic_GAN_Distributed.py](../Examples/Image/GAN/Basic_GAN_Distributed.py) and the [cntk.learners.distributed_multi_learner_test.py](../bindings/python/cntk/learners/tests/distributed_multi_learner_test.py)

## Bug fixes
- Fixed convergence issue in Tutorial 201B
- Fixed pooling/unpooling to support free dimension
- Fixed crash in CNTKBinaryFormat deserializer when crossing sweep boundary
- Fixed shape inference bug in RNN step function for scalar broadcasting
- Fixed a build bug when mpi=no
- Improved distributed training aggregation speed by increasing packing threshold, and expose the knob in V2
- Fixed a memory leak in MKL layout
- Fixed a bug in cntk.convert API in misc.converter.py, which prevents converting complex networks.
