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

## Bug fixes
- Fixed convergence issue in Tutorial 201B
- Fixed pooling/unpooling to support free dimension