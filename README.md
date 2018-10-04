## CNTK

| **Chat** | **Windows build status** | **Linux build status** |
|-------------|-------------|---------------|
| [![Join the chat at https://gitter.im/Microsoft/CNTK](https://badges.gitter.im/Microsoft/CNTK.svg)](https://gitter.im/Microsoft/CNTK?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) | [![Build Status](https://aiinfra.visualstudio.com/_apis/public/build/definitions/a95b3960-90bb-440b-bd18-d3ec5d1cf8c3/126/badge)](https://cntk.ai/nightly-windows.html) | [![Build Status](https://aiinfra.visualstudio.com/_apis/public/build/definitions/a95b3960-90bb-440b-bd18-d3ec5d1cf8c3/127/badge)](https://cntk.ai/nightly-linux.html) |

The Microsoft Cognitive Toolkit (https://cntk.ai) is a unified deep learning toolkit that describes neural networks as a series of computational steps via a directed graph. In this directed graph, leaf nodes represent input values or network parameters, while other nodes represent matrix operations upon their inputs. CNTK allows users to easily realize and combine popular model types such as feed-forward DNNs, convolutional nets (CNNs), and recurrent networks (RNNs/LSTMs). It implements stochastic gradient descent (SGD, error backpropagation) learning with automatic differentiation and parallelization across multiple GPUs and servers. CNTK has been available under an open-source license since April 2015. It is our hope that the community will take advantage of CNTK to share ideas more quickly through the exchange of open source working code.

## Installation

* [Setup CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-your-machine)
    * Windows ([Python-only](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python) / [Script-driven](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-binary-script) / [Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-binary-manual))
    * Linux ([Python-only](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-python) / [Script-driven](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-binary-script) / [Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-linux-binary-manual) / [Docker](https://docs.microsoft.com/en-us/cognitive-toolkit/cntk-docker-containers))
* [CNTK backend for Keras](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras)
* [Setup CNTK development environment](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-development-environment)
    * Windows ([Script-driven](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-with-script-on-windows) / [Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-windows))
    * Linux ([Manual](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-linux))
    
### Installing nightly packages

If you prefer to use latest CNTK bits from master, use one of the CNTK nightly packages:

* [Nightly packages for Windows](https://cntk.ai/nightly-windows.html)
* [Nightly packages for Linux](https://cntk.ai/nightly-linux.html)

## Learning CNTK

You can learn more about using and contributing to CNTK with the following resources:

* [General documentation](https://docs.microsoft.com/en-us/cognitive-toolkit/)
* [Python API documentation](https://cntk.ai/pythondocs/)
* [Evaluation documentation (C++, C#/.NET, Python, Java)](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Evaluation-Overview)
* [Manual](https://github.com/Microsoft/CNTK/tree/master/Manual)
* [Tutorials](https://docs.microsoft.com/en-us/cognitive-toolkit/tutorials)
* [Examples](https://docs.microsoft.com/en-us/cognitive-toolkit/Examples)
* [Pretrained models](./PretrainedModels)
* [Blog](https://www.microsoft.com/en-us/cognitive-toolkit/blog/)
* [Presentations](https://docs.microsoft.com/en-us/cognitive-toolkit/Presentations)
* [License](./LICENSE.md)

## More information

* [Contribute to CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Contributing-to-CNTK)
* [FAQ](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-FAQ)
* [Feedback](https://docs.microsoft.com/en-us/cognitive-toolkit/Feedback-Channels)

## Disclaimer

CNTK is in active use at Microsoft and constantly evolving. There will be bugs.

## Microsoft Open Source Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## News

> You can find more news on [the official project feed](https://docs.microsoft.com/en-us/cognitive-toolkit/news)

### Project changelog
***2018-09-17.*** CNTK 2.6.0
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
- Added CNTK support for ONNX `Unsqueeze` op.

### Bug or minor fixes:
- Updated LRN op to match ONNX 1.2 spec where the `size` attribute has the semantics of diameter, not radius. Added validation if LRN kernel size is larger than channel size.
- Updated `Min`/`Max` import implementation to handle variadic inputs.
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


***2018-04-16.*** CNTK 2.5.1

Repack CNTK 2.5 with third party libraries included in the bundles (Python wheel packages)

---

***2018-03-15.*** CNTK 2.5

Change profiler details output format to be `chrome://tracing`

Enable per-node timing. Working example [here](/Examples/Image/Classification/MLP/Python/SimpleMNIST.py)
* per-node timing creates items in profiler details when profiler is enabled.
* usage in Python:

```python
import cntk as C
C.debugging.debug.set_node_timing(True)
C.debugging.start_profiler() # optional
C.debugging.enable_profiler() # optional
#<trainer|evaluator|function> executions
<trainer|evaluator|function>.print_node_timing()
C.debugging.stop_profiler()
```

Example profiler details view in `chrome://tracing`
![ProfilerDetailWithNodeTiming](https://cntk.ai/Images/ProfilerDetailWithNodeTiming.jpg)

CPU inference performance improvements using MKL
* Accelerates some common tensor ops in Intel CPU inference for float32, especially for fully connected networks
* Can be turned on/off by `cntk.cntk_py.enable_cpueval_optimization()/cntk.cntk_py.disable_cpueval_optimization()`

1BitSGD incorporated into CNTK
* `1BitSGD` source code is now available with CNTK license (MIT license) under `Source/1BitSGD/`
* `1bitsgd` build target was merged into existing gpu target

New loss function: hierarchical softmax
* Thanks @yaochengji for the contribution!

Distributed Training with Mulitple Learners
* Trainer now accepts multiple parameter learners for distributed training. With this change, different parameters of a network can be learned by different learners in a single training session. This also facilitates distributed training for GANs. For more information, please refer to the [Basic_GAN_Distributed.py](/Examples/Image/GAN/Basic_GAN_Distributed.py) and the [cntk.learners.distributed_multi_learner_test.py](/bindings/python/cntk/learners/tests/distributed_multi_learner_test.py)

Operators
* Added `MeanVarianceNormalization` operator. 

Bug fixes
* Fixed convergence issue in Tutorial 201B
* Fixed pooling/unpooling to support free dimension for sequences
* Fixed crash in `CNTKBinaryFormat` deserializer when crossing sweep boundary
* Fixed shape inference bug in RNN step function for scalar broadcasting
* Fixed a build bug when `mpi=no`
* Improved distributed training aggregation speed by increasing packing threshold, and expose the knob in V2
* Fixed a memory leak in MKL layout
* Fixed a bug in `cntk.convert` API in `misc.converter.py`, which prevents converting complex networks.

ONNX
* Updates
    * CNTK exported ONNX models are now `ONNX.checker` compliant. 
    * Added ONNX support for CNTK’s `OptimizedRNNStack` operator (LSTM only).
    * Added support for LSTM and GRU operators
    * Added support for experimental ONNX op `MeanVarianceNormalization`.
    * Added support for experimental ONNX op `Identity`.
    * Added support for exporting CNTK’s `LayerNormalization` layer using ONNX `MeanVarianceNormalization` op.
* Bug or minor fixes:
    * Axis attribute is optional in CNTK’s ONNX `Concat` operator.
    * Bug fix in ONNX broadcasting for scalars.
    * Bug fix in ONNX ConvTranspose operator. 
    * Backward compatibility bug fix in `LeakyReLu` (argument ‘alpha’ reverted to type double).

Misc
* Added a new API `find_by_uid()` under `cntk.logging.graph`. 

---

***2018-02-28.*** CNTK supports nightly build

If you prefer to use latest CNTK bits from master, use one of the CNTK nightly package.
* [Nightly packages for Windows](https://cntk.ai/nightly-windows.html)
* [Nightly packages for Linux](https://cntk.ai/nightly-linux.html)

Alternatively, you can also click corresponding build badge to land to nightly build page.

---

***2018-01-31.* CNTK 2.4**

Highlights:
* Moved to CUDA9, cuDNN 7 and Visual Studio 2017.
* Removed Python 3.4 support.
* Added Volta GPU and FP16 support.
* Better ONNX support.
* CPU perf improvement.
* More OPs.

OPs
* `top_k` operation: in the forward pass it computes the top (largest) k values and corresponding indices along the specified axis. In the backward pass the gradient is scattered to the top k elements (an element not in the top k gets a zero gradient).
* `gather` operation now supports an axis argument
* `squeeze` and `expand_dims` operations for easily removing and adding singleton axes
* `zeros_like` and `ones_like` operations. In many situations you can just rely on CNTK correctly broadcasting a simple 0 or 1 but sometimes you need the actual tensor.
* `depth_to_space`: Rearranges elements in the input tensor from the depth dimension into spatial blocks. Typical use of this operation is for implementing sub-pixel convolution for some image super-resolution models.
* `space_to_depth`: Rearranges elements in the input tensor from the spatial dimensions to the depth dimension. It is largely the inverse of DepthToSpace.
* `sum` operation: Create a new Function instance that computes element-wise sum of input tensors.
* `softsign` operation: Create a new Function instance that computes the element-wise softsign of a input tensor.
* `asinh` operation: Create a new Function instance that computes the element-wise asinh of a input tensor.
* `log_softmax` operation: Create a new Function instance that computes the logsoftmax normalized values of a input tensor.
* `hard_sigmoid` operation: Create a new Function instance that computes the hard_sigmoid normalized values of a input tensor.
* `element_and`, `element_not`, `element_or`, `element_xor` element-wise logic operations
* `reduce_l1` operation: Computes the L1 norm of the input tensor's element along the provided axes.
* `reduce_l2` operation: Computes the L2 norm of the input tensor's element along the provided axes.
* `reduce_sum_square` operation: Computes the sum square of the input tensor's element along the provided axes.
* `image_scaler` operation: Alteration of image by scaling its individual values.

ONNX
* There have been several improvements to ONNX support in CNTK.
* Updates
  * Updated ONNX `Reshape` op to handle `InferredDimension`.
  * Adding `producer_name` and `producer_version` fields to ONNX models.
  * Handling the case when neither `auto_pad` nor `pads` atrribute is specified in ONNX `Conv` op.
* Bug fixes
  * Fixed bug in ONNX `Pooling` op serialization
  * Bug fix to create ONNX `InputVariable` with only one batch axis.
  * Bug fixes and updates to implementation of ONNX `Transpose` op to match updated spec.
  * Bug fixes and updates to implementation of ONNX `Conv`, `ConvTranspose`, and `Pooling` ops to match updated spec.

Operators
* Group convolution
  * Fixed bug in group convolution. Output of CNTK `Convolution` op will change for groups > 1. More optimized implementation of group convolution is expected in the next release.
  * Better error reporting for group convolution in `Convolution` layer.

Halide Binary Convolution
- The CNTK build can now use optional [Halide](http://halide-lang.org/) libraries to build `Cntk.BinaryConvolution.so/dll` library that can be used with the `netopt` module. The library contains optimized binary convolution operators that perform better than the python based binarized convolution operators. To enable Halide in the build, please download [Halide release](https://github.com/halide/Halide/releases) and set `HALIDE_PATH` environment varibale before starting a build. In Linux, you can use `./configure --with-halide[=directory]` to enable it. For more information on how to use this feature, please refer to [How_to_use_network_optimization](https://github.com/Microsoft/CNTK/blob/master/Manual/Manual_How_to_use_network_optimizations.ipynb).

See more in the [Release Notes](https://docs.microsoft.com/en-us/cognitive-toolkit/ReleaseNotes/CNTK_2_4_Release_Notes).
Get the Release from the [CNTK Releases page](https://github.com/Microsoft/CNTK/releases).

---

***2018-01-22.*** CNTK support for CUDA 9

CNTK now supports CUDA 9/cuDNN 7. This requires an update to build environment to Ubuntu 16/GCC 5 for Linux, and Visual Studio 2017/VCTools 14.11 for Windows. With CUDA 9, CNTK also added a preview for 16-bit floating point (a.k.a FP16) computation.

Please check out the example of FP16 in ResNet50 [here](./Examples/Image/Classification/ResNet/Python/TrainResNet_ImageNet_Distributed.py)

Notes on FP16 preview:
* FP16 implementation on CPU is not optimized, and it's not supposed to be used in CPU inference directly. User needs to convert the model to 32-bit floating point before running on CPU.
* Loss/Criterion for FP16 training needs to be 32bit for accumulation without overflow, using cast function. Please check the example above.
* Readers do not have FP16 output unless using numpy to feed data, cast from FP32 to FP16 is needed. Please check the example above.
* FP16 gradient aggregation is currently only implemented on GPU using NCCL2. Distributed training with FP16 with MPI is not supported.
* FP16 math is a subset of current FP32 implementation. Some model may get Feature Not Implemented exception using FP16.
* FP16 is currently not supported in BrainScript. Please use Python for FP16.

To setup build and runtime environment on Windows:
* Install [Visual Studio 2017](https://www.visualstudio.com/downloads/) with following workloads and components. From command line (use Community version installer as example):
    `vs_community.exe --add Microsoft.VisualStudio.Workload.NativeDesktop --add Microsoft.VisualStudio.Workload.ManagedDesktop --add Microsoft.VisualStudio.Workload.Universal --add Microsoft.Component.PythonTools --add Microsoft.VisualStudio.Component.VC.Tools.14.11`
* Install [NVidia CUDA 9](https://developer.nvidia.com/cuda-90-download-archive?target_os=Windows&target_arch=x86_64)
* From PowerShell, run:
    [DevInstall.ps1](./Tools/devInstall/Windows/DevInstall.ps1)
* Start VCTools 14.11 command line, run:
    `cmd /k "%VS2017INSTALLDIR%\VC\Auxiliary\Build\vcvarsall.bat" x64 --vcvars_ver=14.11`
* Open [CNTK.sln](./CNTK.sln) from the VCTools 14.11 command line. Note that starting CNTK.sln other than VCTools 14.11 command line, would causes CUDA 9 [build error](https://developercommunity.visualstudio.com/content/problem/163758/vs-2017-155-doesnt-support-cuda-9.html).

To setup build and runtime environment on Linux using docker, please build Unbuntu 16.04 docker image using Dockerfiles [here](./Tools/docker). For other Linux systems, please refer to the Dockerfiles to setup dependent libraries for CNTK.

