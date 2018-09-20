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

## Sequential Convolution
The implementation of sequential convolution in CNTK has been updated. The updated implementation creates a separate sequential convolution layer. Different from regular convolution layer, this operation convolves also on the dynamic axis(sequence), and filter_shape[0] is applied to that axis. The updated implementation supports broader cases, such as where stride > 1 for the sequence axis.

For example, a sequential convolution over a batch of one-channel black-and-white images. The images have the same fixed height of 640, but each with width of variable lengths. The width is then represented by sequential axis. Padding is enabled, and strides for both width and height are 2.

     >>> f = SequentialConvolution((3,3), reduction_rank=0, pad=True, strides=(2,2), activation=C.relu)
     >>> x = C.input_variable(**Sequence[Tensor[640]])
     >>> x.shape
         (640,)
     >>> h = f(x)
     >>> h.shape
         (320,)
     >>> f.W.shape
         (1, 1, 3, 3)

## Operators
### depth_to_space and space_to_depth
There is a breaking change in the **depth_to_space** and **space_to_depth** operators. These have been updated to match ONNX specification, specifically
the permutation for how the depth dimension is placed as blocks in the spatial dimensions, and vice-versa, has been changed. Please refer to the updated doc
examples for these two ops to see the change.

### Tan and Atan
Added support for trigonometric ops `Tan` and `Atan`.

### ELU
Added support for `alpha` attribute in ELU op.

### Convolution
Updated auto padding algorithms of `Convolution` to produce symmetric padding at best effort on CPU, without affecting the final convolution output values. This update increases the range of cases that could be covered by MKL API and improves the performance, E.g. ResNet50.

## Default arguments order
There is a breaking change in the **arguments** property in CNTK python API. The default behavior has been updated to return arguments in python order instead of in C++ order. This way it will return arguments in the same order as they are fed into ops. If you wish to still get arguments in C++ order, you can simply override the global option. This change should only affect the following ops: Times, TransposeTimes, and Gemm(internal). 

## Bug fixes
- Updated doc for Convolution layer to include group and dilation arguments.
- Added improved input validation for group convolution.
- Updated `LogSoftMax` to use more numerically stable implementation.
- Fixed Gather op's incorrect gradient value.
- Added validation for 'None' node in python clone substitution.
- Added validation for padding channel axis in convolution.
- Added CNTK native default lotusIR logger to fix the "Attempt to use DefaultLogger" error when loading some ONNX models.
- Added proper initialization for ONNX TypeStrToProtoMap.
- Updated python doctest to handle different print format for newer version numpy(version >= 1.14).
- Fixed Pooling(CPU) to produce correct output values when kernel center is on padded input cells.

## ONNX
### Updates
- Updated CNTK's ONNX import/export to use ONNX 1.2 spec.
- Major update to how batch and sequence axes are handled in export and import. As a result, the complex scenarios and edge cases are handled accurately.
- Added support for exporting and importing Float16 models.
- Updated CNTK's ONNX `BatchNormalization` op export/import to latest spec.
- Added model domain to ONNX model export.
- Improved error reporting during import and export of ONNX models.
- Updated `DepthToSpace` and `SpaceToDepth` ops to match ONNX spec on the permutation for how the depth dimension is placed as block dimension.
- Added support for exporting `alpha` attribute in `ELU` ONNX op.
- Major overhaul to `Convolution` and `Pooling` export. Unlike before, these ops do not export an explicit `Pad` op in any situation.
- Major overhaul to `ConvolutionTranspose` export and import. Attributes such as `output_shape`, `output_padding`, and `pads` are fully supported.
- Added support for CNTK's `StopGradient` as a no-op.
- Added ONNX support for TopK op.
- Added ONNX support for sequence ops: sequence.slice, sequence.first, sequence.last, sequence.reduce_sum, sequence.reduce_max, sequence.softmax. For these ops, there is no need to expand ONNX spec. CNTK ONNX exporter just builds computation equavalent graphs for these sequence ops.
- Added full support for Softmax op.
- Made CNTK broadcast ops compatible with ONNX specification.
- Handle to_batch, to_sequence, unpack_batch, sequence.unpack ops in CNTK ONNX exporter.
- ONNX tests to export ONNX test cases for other toolkits to run and to validate.
- Fixed `Hardmax`/`Softmax`/`LogSoftmax` import/export.
- Added support for `Select` op export.
- Added import/export support for several trigonometric ops.
- Updated CNTK support for ONNX `MatMul` op.
- Updated CNTK support for ONNX `Gemm` op.
- Updated CNTK's ONNX `MeanVarianceNormalization` op export/import to latest spec.
- Updated CNTK's ONNX `LayerNormalization` op export/import to latest spec.
- Updated CNTK's ONNX `PRelu` op export/import to latest spec.
- Updated CNTK's ONNX `Gather` op export/import to latest spec.
- Updated CNTK's ONNX `ImageScaler` op export/import to latest spec.
- Updated CNTK's ONNX `Reduce` ops export/import to latest spec.
- Updated CNTK's ONNX `Flatten` op export/import to latest spec.
- Added support for exporting CNTK's `TimesTranspose` op to ONNX.
- Added CNTK support for ONNX `Unsqueeze` op.

### Bug or minor fixes:
- Updated LRN op to match ONNX 1.2 spec where the `size` attribute has the semantics of diameter, not radius. Added validation if LRN kernel size is larger than channel size.
- Updated `Min`/`Max` import implementation to handle variadic inputs.
- Fixed CNTK's `Times` op export to handle the scenario where only one input has batch axis.
- Fixed possible file corruption when resaving on top of existing ONNX model file.

## .Net Support
The Cntk.Core.Managed library has officially been converted to .Net Standard and supports .Net Core and .Net Framework applications on both Windows and Linux. Starting from this release, .Net developers should be able to restore CNTK Nuget packages using new .Net SDK style project file with package management format set to PackageReference.

The following C# code now works on both Windows and Linux:

     >>> var weightParameterName = "weight";
	 >>> var biasParameterName = "bias";
	 >>> var inputName = "input";
	 >>> var outputDim = 2;
	 >>> var inputDim = 3;
	 >>> Variable inputVariable = Variable.InputVariable(new int[] { inputDim }, DataType.Float, inputName);
	 >>> var weightParameter = new Parameter(new int[] { outputDim, inputDim }, DataType.Float, 1, device, weightParameterName);
	 >>> var biasParameter = new Parameter(new int[] { outputDim }, DataType.Float, 0, device, biasParameterName);
	 >>> 
     >>> Function modelFunc = CNTKLib.Times(weightParameter, inputVariable) + biasParameter;

For example, simply adding an ItemGroup clause in the .csproj file of a .Net Core application is sufficient:
     >>> <Project Sdk="Microsoft.NET.Sdk">
     >>>
     >>>   <PropertyGroup>
     >>>     <TargetFramework>netcoreapp2.1</TargetFramework>
     >>>     <Platforms>x64</Platforms>
     >>>   </PropertyGroup>
     >>>
     >>>   <ItemGroup>
     >>>     <PackageReference Include="CNTK.GPU" Version="2.6.0" />
     >>>   </ItemGroup>
     >>>
     >>> </Project>

### Bug or minor fixes:
- Fixed C# string and char to native wstring and wchar UTF conversion issues on Linux.
- Fixed multibyte and wide character conversions across the codebase.
- Fixed Nuget package mechanism to pack for .Net Standard.
- Fixed a memory leak issue in Value class in C# API where Dispose was not called upon object destruction.

## Misc

