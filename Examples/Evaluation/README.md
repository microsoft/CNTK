#Eval Examples

The folder contains examples using the CNTK evaluation library.
-CPPEvalClient: it is C++ example code and VS C++ project file used by cntk.sln
-CPPEvalExample: It contains only VS C++ project file used in the CNTK binary download package. It shares the same C++ code in CPPEvalClient.
-CSEvalClient: it is C# example code and VS C# project file used by cntk.sln.
-CSEvalExample: it contains only VS C# project file used in the CNTK binary download package. The project uses Eval Nuget package. It shares the same C# code in CSEvalClient.
-EvalExample.sln: the VS solution file to build examples in the CNTK binary download package. It creates 2 binaries in the directory $(SolutionDir)..\..\cntk\EvalExample.$(Configuration)\:
   *CPPEvalExample.exe: the C++ example executable. To start the example, please first include the directory containing CNTK dependent dlls, usually $(SolutionDir)..\..\cntk, in the PATH environment variable.
   *CSEvalExample.exe: the C# example executable
  