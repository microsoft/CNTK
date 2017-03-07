@ECHO OFF
SET SOL_DIR=%~f1
SET BUILD_TYPE=%2

SET START_DIR=%CD%
SET START_PATH=%PATH%
SET PATH=%PATH%;%SOL_DIR%\x64\%BUILD_TYPE%
cd "%SOL_DIR%\bindings\java\JavaEvalTest"

"%JAVA_HOME%\bin\javac" -cp "%SOL_DIR%\bindings\java\SwigProxyClasses\cntk.jar" src\Main.java
ECHO
"%JAVA_HOME%\bin\java" -Djava.library.path=%SOL_DIR%/x64/%BUILD_TYPE% -classpath "%SOL_DIR%\bindings\java\JavaEvalTest\src;%SOL_DIR%\bindings\java\SwigProxyClasses\cntk.jar" Main

@ECHO OFF
SET PATH=%START_PATH% 
cd %START_DIR%
ECHO