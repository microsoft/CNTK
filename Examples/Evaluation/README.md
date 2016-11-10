#EvalClients

The folder contains some examples using the CNTK evaluation library. Please note that only the 64-bit target is supported by CNTK evaluation library.

-CPPEvalClient: demonstrate the use of the C++ CNTK eval lib. Only the release configuration is supported.  

-CSEvalClient: demonstrate the use of the C# CNTK eval lib.

-EvalClients.sln: the VS2013 solution file to build examples. It creates two binaries in the directory $(SolutionDir)..\..\x64\:

    - CPPEvalClient.$(Configuration)\CPPEvalClient.exe: the C++ example executable. To run the example, please first include the directory containing CNTK dependent dlls, usually $(SolutionDir)..\..\cntk, in the PATH environment variable. 
    
    - CSEvalClient.$(Configuration)\CSEvalClient.exe: the C# example executable.
