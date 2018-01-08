# CNTK v2.4 Release Notes

## Highlights of this Release
- Move to CUDA9, cuDNN 7 and Visual Studio 2017.
- Support Volta GPU and FP16.
- Better ONNX support.
- CPU perf improvement.
- More OPs.

## OPs
- ``top_k`` operation: in the forward pass it computes the top (largest) k values and corresponding indices along the specified axis. In the backward pass the gradient is scattered to the top k elements (an element not in the top k gets a zero gradient).
- ``gather`` operation now supports an axis argument
- ``squeeze`` and ``expand_dims`` operations for easily removing and adding singleton axes
- ``zeros_like`` and ``ones_like`` operations. In many situations you can just rely on CNTK correctly broadcasting a simple 0 or 1 but sometimes you need the actual tensor.

## ONNX
- Improved ONNX support in CNTK.
- Update ONNX to the latest ONNX from https://github.com/onnx/onnx
- Fixed several bugs.

