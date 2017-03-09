SET SOL_DIR=%~f1
%SWIG_PATH%\swig.exe -c++ -java -DMSC_VER -I%SOL_DIR%Source\CNTKv2LibraryDll\API -I%SOL_DIR%bindings\common -package com.microsoft.CNTK -outdir  %SOL_DIR%bindings\java\SwigProxyClasses\com\microsoft\CNTK  %SOL_DIR%bindings\java\Swig\cntk_java.i
pushd %SOL_DIR%\bindings\java\SwigProxyClasses
javac .\com\microsoft\CNTK\*.java
jar -cvf cntk.jar .\com\microsoft\CNTK\*.class
popd
