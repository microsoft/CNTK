REM @ECHO OFF
SET SOL_DIR=%~f1
SET BUILD_PATH=%~f2

SET START_DIR=%CD%
SET START_PATH=%PATH%
SET PATH=%PATH%;%BUILD_PATH%
cd "%SOL_DIR%\bindings\java\JavaEvalTest"

"%JAVA_HOME%\bin\javac" -cp "%SOL_DIR%\bindings\java\SwigProxyClasses\cntk.jar" src\Main.java
REM ECHO
"%JAVA_HOME%\bin\java" -Djava.library.path=%BUILD_PATH% -classpath "%SOL_DIR%\bindings\java\JavaEvalTest\src;%SOL_DIR%\bindings\java\SwigProxyClasses\cntk.jar" Main

REM @ECHO OFF
SET PATH=%START_PATH% 
cd %START_DIR%
REM ECHO