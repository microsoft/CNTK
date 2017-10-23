setlocal enableextensions enabledelayedexpansion
set project_dir=%~f1

echo Generating Java binding...
echo The project directory is "%project_dir%"

if not exist "%project_dir%com\microsoft\CNTK\" mkdir "%project_dir%com\microsoft\CNTK\"
"%SWIG_PATH%\swig.exe" -c++ -java -D_MSC_VER -Werror -I"%project_dir%..\..\..\Source\CNTKv2LibraryDll\API" -I"%project_dir%..\..\common" -package com.microsoft.CNTK -outdir  "%project_dir%com\microsoft\CNTK"  "%project_dir%cntk_java.i" || (
  echo Running SWIG for Java binding failed!
  exit /B 1
)



