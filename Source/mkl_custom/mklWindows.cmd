@echo off
echo.
echo This batch file will build a custom MKL dynamic link library for usage by CNTK.
echo.
echo Requirements:
echo  - Intel MKL SDK installed on the machine
echo  - MKLROOT environment variable is set to the MKL directory inside the Intel MKL SDK
echo  - Visual Studio 2013 installed and included in the path
echo.

if "%MKLROOT%" == "" (
  echo Error: Environment variable MKLROOT is undefined
  exit /b 1
)

if not exist "%MKLROOT%" ( 
  echo Error: Directory doesn't exist: "%MKLROOT%"
  exit /b 1
)

where /q nmake.exe
if errorlevel 1 (
  echo Error: NMAKE.EXE not in path
  exit /b 1
)

where /q link.exe
if errorlevel 1 (
  echo Error: LINK.EXE not in path
  exit /b 1
)

echo Copying Builder-LIB directory
xcopy /s /e /y /i "%MKLROOT%\tools\builder\lib" lib 

echo.
echo Calling NMAKE libintel64 export=cntklist.txt MKLROOT="%MKLROOT%"
NMAKE /f "%MKLROOT%\tools\builder\makefile" libintel64 export=cntklist.txt MKLROOT="%MKLROOT%"

if errorlevel 1 (
  echo Error: NMAKE.exe failed.
  exit /b 1
)

echo.
echo Removing LIB directory
rmdir /s /q lib

set /p CURRENTVER=<version.txt

echo.
echo Copying into Publish\%CURRENTVER%

rmdir /s /q Publish
md Publish\%CURRENTVER%\x64
md Publish\%CURRENTVER%\include

move mkl_custom.dll Publish\%CURRENTVER%\x64
move mkl_custom.lib Publish\%CURRENTVER%\x64
del mkl_custom.exp
del *.manifest
copy "%MKLROOT%\..\redist\intel64_win\compiler\libiomp5md.dll" Publish\%CURRENTVER%\x64

rem MKL Include files needed to build CNTK

for %%h in (
  mkl_blas.h
  mkl_cblas.h
  mkl_df_defines.h
  mkl_df_functions.h
  mkl_df_types.h
  mkl_df.h
  mkl_dfti.h
  mkl_direct_call.h
  mkl_dss.h
  mkl_lapack.h
  mkl_lapacke.h
  mkl_pardiso.h
  mkl_poisson.h
  mkl_rci.h
  mkl_service.h
  mkl_solvers_ee.h
  mkl_sparse_handle.h
  mkl_spblas.h
  mkl_trans.h
  mkl_trig_transforms.h
  mkl_types.h
  mkl_version.h
  mkl_vml_defines.h
  mkl_vml_functions.h
  mkl_vml_types.h
  mkl_vml.h
  mkl_vsl_defines.h
  mkl_vsl_functions.h
  mkl_vsl_types.h
  mkl_vsl.h
  mkl.h
) do (
  copy "%MKLROOT%\include\%%h" Publish\%CURRENTVER%\include
)

copy license.txt Publish\%CURRENTVER%
