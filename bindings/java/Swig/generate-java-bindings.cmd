cd C:\repos\cntk\bindings\java\Swig
swig -c++ -java -IC:\repos\cntk\Source\CNTKv2LibraryDll\API -IC:\repos\cntk\bindings\common -package com.microsoft.CNTK -outdir  $C:\repos\cntk\bindings\java\CNTKLibraryManagedDll\SwigProxyClasses cntk_java.i
cd C:\repos\cntk\bindings\java\CNTKLibraryManagedDll\SwigProxyClasses
javac *.java
del *.java
jar -cvf cntk.jar *.class
del *.class
cd C:\repos\cntk\bindings\java\Swig
