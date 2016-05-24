rem @echo off
echo.
echo This batch file will build a custom MKL dynamic link library for usage by CNTK.
echo.
echo Requirements:
echo  - Intel MKL SDK installed on the machine
echo  - MKLROOT environment variable is set to the MKL directory inside the Intel MKL SDK
echo  - Visual Studio 2013 installed and included in the path ()
echo.

if "%MKLROOT%" == "" (
  echo Error:  Environment variable MKLROOT is undefined
  exit /b 1
)
if not exist "%MKLROOT%" ( 
  echo Error: Directory doesn't exist: "%MKLROOT%"
  exit /b 1
)

where /q nmake.exe
if %errorlevel% GTR 0  (
  echo Error: NMAKE.EXE not in path
  exit /b 1
)

where /q link.exe
if %errorlevel% GTR 0  (
  echo Error: LINK.EXE not in path
  exit /b 1
)

echo Copying Builder-LIB directory
xcopy /s /e /y /i "%MKLROOT%\tools\builder\lib" lib 

echo.
echo Calling NMAKE libintel64 export=cntklist.txt MKLROOT="%MKLROOT%"
NMAKE /f "%MKLROOT%\tools\builder\makefile" libintel64 export=cntklist.txt MKLROOT="%MKLROOT%"

if %errorlevel% GTR 0  (
  echo Error: LINK.EXE not in path
  exit /b 1
)

echo.
echo Removing LIB directory
del /q lib
rd lib

dir version.txt

set /p CURRENTVER=<VERSION.TXT

echo.
echo Copying into Publish\%CURRENTVER%


md Publish
md Publish\%CURRENTVER%
md Publish\%CURRENTVER%\x64
md Publish\%CURRENTVER%\include

move mkl_custom.dll Publish\%CURRENTVER%\x64
move mkl_custom.lib Publish\%CURRENTVER%\x64
del  mkl_custom.exp
del  *.manifest
copy "%MKLROOT%\..\redist\intel64_win\compiler\libiomp5md.dll"  Publish\%CURRENTVER%\x64

rem MKL Include files needed to build CNTK
rem
copy "%MKLROOT%\include\mkl.h"             Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_version.h"     Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_types.h"       Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_blas.h"        Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_trans.h"       Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_cblas.h"       Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_spblas.h"      Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_lapack.h"      Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_lapacke.h"     Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_pardiso.h"     Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_dss.h"         Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_sparse*.h"     Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_rci.h"         Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_service.h"     Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_vsl*.h"        Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_vml*.h"        Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_df*.h"         Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_trig*.h"       Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_poisson.h"     Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_solvers_ee.h"  Publish\%CURRENTVER%\include
copy "%MKLROOT%\include\mkl_direct_call.h" Publish\%CURRENTVER%\include

copy license.txt  Publish\%CURRENTVER%


exit /b 0







