SET SOL_DIR=%~dpf1
SET BUILD_PATH=%~dpf2

SET START_PATH=%PATH%
SET PATH=%PATH%;%BUILD_PATH%

pushd "%SOL_DIR%\bindings\java\JavaEvalTest"
"%JAVA_HOME%\bin\javac" -cp "%SOL_DIR%\bindings\java\SwigProxyClasses\cntk.jar" src\Main.java || (
  echo "Java Compilation Failed"
  popd
  SET PATH=%START_PATH% 
  EXIT /B 1
)
"%JAVA_HOME%\bin\java" -Djava.library.path=%BUILD_PATH% -classpath "%SOL_DIR%\bindings\java\JavaEvalTest\src;%SOL_DIR%\bindings\java\SwigProxyClasses\cntk.jar" Main || (
  echo "Running Java Failed"
  popd
  SET PATH=%START_PATH% 
  EXIT /B 1
)
popd
SET PATH=%START_PATH% 