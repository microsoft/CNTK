SET SOL_DIR=%~f1
if not exist %SOL_DIR%\bindings\java\Swig\com\microsoft\CNTK\ mkdir %SOL_DIR%\bindings\java\Swig\com\microsoft\CNTK\
%SWIG_PATH%\swig.exe -c++ -java -DMSC_VER -I%SOL_DIR%Source\CNTKv2LibraryDll\API -I%SOL_DIR%bindings\common -package com.microsoft.CNTK -outdir  %SOL_DIR%bindings\java\Swig\com\microsoft\CNTK  %SOL_DIR%bindings\java\Swig\cntk_java.i

pushd %SOL_DIR%\bindings\java\Swig
REM: TODO: add check whether javac/jar exist.
%JAVA_HOME%\bin\javac .\com\microsoft\CNTK\*.java
%JAVA_HOME%\bin\jar -cvf cntk.jar .\com\microsoft\CNTK\*.class
rd com /q /s
popd

