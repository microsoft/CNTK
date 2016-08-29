#EvalClients

Thefolder contains examples using the CNTK evaluation library. Please note that only 64-bit target is supported by CNTK evaluation library.
-CPPEvalClient:it is C++ example code and VS C++ project file
-CSEvalClient:it is C# example code and VS C# project file
-EvalClients.sln:the VS solution file to build examples in the CNTK binary download package. It creates 2 binaries in the directory $(SolutionDir)..\..\x64\:
*CPPEvalClient.$(Configuration)\CPPEvalClient.exe: the C++ example executable. Only the release configuraiton is supported by the CNTK binary download. To start the example, please first include the directory containing CNTK dependent dlls, usually $(SolutionDir)..\..\cntk, in the PATH environment variable.  
*CSEvalClient.$(Configuration)\CSEvalClient.exe: the C# example executable.
