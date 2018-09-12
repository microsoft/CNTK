# CNTK Current Iteration

## Efficient group convolution
The implementation of group convolution in CNTK has been updated. The updated implementation moves away from creating a sub-graph for group convolution (using slicing and splicing), and instead uses cuDNN7 and MKL2017 APIs directly. This improves the experience both in terms of performance and model size. 

As an example, for a single group convolution op with the following attributes:

- Input tensor (C, H, W) = (32, 128, 128)
- Number of output channels = 32 (channel multiplier is 1)
- Groups = 32 (depth wise convolution)
- Kernel size = (5, 5)

The comparison numbers for this single node are as follows:

| First Header  | GPU exec. time (in millisec., 1000 run avg.) | CPU exec. time (in millisec., 1000 run avg.) | Model Size (in KB, CNTK format)
| ------------- | ------------- | ------------- | ------------- |
| Old implementation  | 9.349  | 41.921  | 38  |
| New implementation  | 6.581  | 9.963  | 5  |
| Speedup/savings   Approx.  | 30%  Approx.  | 65-75%   Approx.  | 87% |

## Operators
### depth_to_space and space_to_depth
There is a breaking change in the **depth_to_space** and **space_to_depth** operators. These have been updated to match ONNX specification, specifically
the permutation for how the depth dimension is placed as blocks in the spatial dimensions, and vice-versa, has been changed. Please refer to the updated doc
examples for these two ops to see the change.

### ELU
Added support for `alpha` attribute in ELU op.

## Default arguments order
There is a breaking change in the **arguments** property in CNTK python API. The default behavior has been updated to return arguments in python order instead of in C++ order. This way it will return arguments in the same order as they are fed into ops. If you wish to still get arguments in C++ order, you can simply override the global option. This change should only affect the following ops: Times, TransposeTimes, and Gemm(internal). 

## Bug fixes
- Updated doc for Convolution layer to include group and dilation arguments.
- Added improved input validation for group convolution.


## ONNX
### Updates
- Major update to how batch and sequence axes are handled in export and import. As a result, the complex scenarios and edge cases are handled accurately.
- Updated CNTK's ONNX `BatchNormalization` op export/import to latest spec.
- Added model domain to ONNX model export.
- Improved error reporting during import and export of ONNX models.
- Updated `DepthToSpace` and `SpaceToDepth` ops to match ONNX spec on the permutation for how the depth dimension is placed as block dimension.
- Added support for exporting `alpha` attribute in `ELU` ONNX op.
- Major overhaul to `Convolution` and `Pooling` export. Unlike before, these ops do not export an explicit `Pad` op in any situation.
- Major overhaul to `ConvolutionTranspose` export and import. Attributes such as `output_shape`, `output_padding`, and `pads` are fully supported.
- Added support for CNTK's `StopGradient` as a no-op.


### Bug or minor fixes:
- Updated LRN op to match ONNX 1.2 spec where the `size` attribute has the semantics of diameter, not radius.

## Misc

