#EvalClients

The folder contains some examples using the CNTK evaluation library. Please note that only the 64-bit target is supported by CNTK evaluation library.
The [CNTK Eval Examples](https://github.com/Microsoft/CNTK/wiki/CNTK-Eval-Examples) page provides more details of these examples.

-CPPEvalClient: demonstrate the use of the C++ CNTK EvalDll evaluation API. Only the release configuration is supported.  

-CPPEvalExtendedClient: demonstrate the use of the C++ EvalDll EvalExtended API to evaluate a LSTM model. Only the release configuration is supported.

-CSEvalClient: demonstrate the use of the C# CNTK EvalDll Nuget Package.

-EvalClients.sln: the VS2015 solution file to build examples. It creates two binaries in the directory $(SolutionDir)..\..\x64\:

    - CPPEvalClient.$(Configuration)\CPPEvalClient.exe: To run the example, please first include the directory containing CNTK dependent dlls, usually $(SolutionDir)..\..\cntk, in the PATH environment variable. 

    - CPPEvalExtendedClient.$(Configuration)\CPPEvalExtendedClient.exe: To run the example, please first include the directory containing CNTK dependent dlls, usually $(SolutionDir)..\..\cntk, in the PATH environment variable. 
    
    - CSEvalClient.$(Configuration)\CSEvalClient.exe: the C# example executable.
