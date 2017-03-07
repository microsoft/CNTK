SET SOL_DIR=%~f1
echo %SOL_DIR%
SET START_DIR=%CD%
echo %START_DIR%
%SWIG_PATH%\swig.exe -c++ -java -DMSC_VER -I%SOL_DIR%Source\CNTKv2LibraryDll\API -I%SOL_DIR%bindings\common -package com.microsoft.CNTK -outdir  %SOL_DIR%bindings\java\SwigProxyClasses\com\microsoft\CNTK  %SOL_DIR%bindings\java\Swig\cntk_java.i
cd %SOL_DIR%\bindings\java\SwigProxyClasses
javac .\com\microsoft\CNTK\*.java
jar -cvf cntk.jar .\com\microsoft\CNTK\*.class
cd %START_DIR%
