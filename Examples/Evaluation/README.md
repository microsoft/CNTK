#EvalClients

The folder contains some examples using the CNTK to evaluate a trained model in your application. Please note that Visual Studio 2015 update 3 is required, and only the 64-bit target is supported.

The [CNTK Eval Examples](https://docs.microsoft.com/en-us/cognitive-toolkit/CNTK-Eval-Examples) page provides more details of these examples.

# CNTK Library Eval C++/C# Examples

The CNTKLibraryEvalExamples.sln contains code samples demonstrating how to use the CNTK Library API in C++ and C#.

* CNTKLibraryCSEvalCPUOnlyExamples uses the CNTK Library CPU-Only Nuget package to evaluate models on CPU-only devices in C#.
* CNTKLibraryCSEvalGPUExamples uses the CNTK Library GPU Nuget package to evaluate models on GPU devices in C#.
* CNTKLibraryCPPEvalCPUOnlyExamples uses the CNTK Library C++ API to evaluate models on CPU-only devices. It uses the CNTK Library CPU-Only Nuget package.
* CNTKLibraryCPPEvalGPUExamples uses the CNTK Library C++ API to evaluate models on GPU devices. It uses the CNTK Library GPU Nuget package.

After a successful build, the executable is saved under the $(SolutionDir)..\..$(Platform)$(ProjectName).$(Configuration)\ folder, e.g. ..\..\X64\CNTKLibraryCSEvalCPUOnlyExamples.Release\CNTKLibraryCSEvalCPUOnlyExamples.exe.
On Linux, only C++ is supported. Please refer to Makefile for building samples. The target name CNTKLIBRARY_CPP_EVAL_EXAMPLES is used to build CNTKLibraryCPPEvalExamples.

# Legacy EvalDll C++/C# Examples

Prior to the CNTK 2.0 version, the CNTK EvalDLL was used to evaluate model trained by using cntk.exe with BrainScript. The EvalDLL is still supported, but works only for the model created by cntk.exe with BrainScript. It can not be used to evaluate models created by CNTK 2.0 using Python. We strongly recommend to use the CNTK 2.0 Libraries for evaluation, as it provides more features.

The EvalClients.sln inside LegacyEvalDll folder contains the following projects demonstrating how to use the EvalDll library in C++ and C#.

* CPPEvalClient: this sample uses the C++ EvalDll.
* CPPEvalExtendedClient: this sample uses the C++ extended Eval interface in EvalDll to evaluate a RNN model.
* CSEvalClient: this sample uses the C# EvalDll (only for Windows). It uses the CNTK EvalDll Nuget Package.

After a successful build, the executable is saved under the $(SolutionDir)..\..$(Platform)$(ProjectName).$(Configuration)\ folder, e.g. ..\..\X64\CPPEvalClient.Release\CppEvalClient.exe.
On Linux, please refer to Makefile for building samples. The target name EVAL_CLIENT, and EVAL_EXTENDED_CLIENT are used to build these projects.