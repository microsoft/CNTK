@echo off
setlocal enableDelayedexpansion

::: Copyright (c) Microsoft. All rights reserved.
:::
::: Licensed under the MIT license. See LICENSE.md file in the project root 
::: for full license information.
::: ==============================================================================
:::
::: This is called as a pre-build step for the CNTK executable, taking parameters below.
::: It creates Generated\Windows\buildinfo.h, which makes version information available to the executable itself.

:: Delete the old versions of the buildinfo file, as they can break the build in some scenarios if left hanging around
if exist "..\CNTK\buildinfo.h" del "..\CNTK\buildinfo.h"
if exist "buildinfo.h" del "buildinfo.h"

if not exist "Generated\Windows" mkdir "Generated\Windows"
set _outfile=Generated\Windows\buildinfo.h

:: Grab the parameters
::
:: Note: don't rely on environment variables, since properties may have been
:: overridden at msbuild invocation. By convention, we let parameters start with p_, locals with l_.
:: A Vim search for [%!]\([lp]_\)\@!\w\+[%!:] should only match
:: well-known (non-CNTK-specific) environment variables.
set p_Configuration=%~1
set p_CNTK_MKL_SEQUENTIAL=%~2
set p_CudaPath=%~3
set p_CUDNN_PATH=%~4
set p_CUB_PATH=%~5
set p_CNTK_ENABLE_ASGD=%~6

echo #ifndef _BUILDINFO_H > %_outfile%$$
echo #define _BUILDINFO_H >> %_outfile%$$

FOR /F %%i IN ('hostname') DO SET HOST=%%i
:: assuming hostname always exists

:: note: we'll only use git which is in path
where -q git
if not errorlevel 1 (
    call git --version > NUL 2>&1
    if not errorlevel 1 (
        echo #define _GIT_EXIST >> %_outfile%$$
        FOR /F %%i IN ('call git rev-parse --abbrev-ref HEAD') DO SET l_BRANCH=%%i
        FOR /F %%i IN ('call git rev-parse HEAD') DO SET l_COMMIT=%%i
        set l_STATUS=
        call git diff --quiet --cached
        if not errorlevel 1 call git diff --quiet
        if errorlevel 1 set l_STATUS= ^(modified^)
        echo #define _BUILDBRANCH_  "!l_BRANCH!"      >> %_outfile%$$
        echo #define _BUILDSHA1_    "!l_COMMIT!!l_STATUS!">> %_outfile%$$
    )
)

if "%p_CNTK_MKL_SEQUENTIAL%" == "1" (
  echo #define _MATHLIB_ "mkl-sequential">> %_outfile%$$
) else (
  echo #define _MATHLIB_ "mkl">> %_outfile%$$
)

echo #define _BUILDER_ "%USERNAME%"     >> %_outfile%$$
echo #define _BUILDMACHINE_ "%HOST%"    >> %_outfile%$$

set l_scriptpath=%~dp0
set l_buildpath="%l_scriptpath:\=\\%"
echo #define _BUILDPATH_    %l_buildpath%     >> %_outfile%$$

set l_build_type=Unknown
set l_build_target=Unknown
:: Configuration property provided by CNTK.vcxproj
if /i "%p_Configuration%" == "Debug" set l_build_type=Debug&set l_build_target=GPU
if /i "%p_Configuration%" == "Debug_CpuOnly" set l_build_type=Debug&set l_build_target=CPU-only
if /i "%p_Configuration%" == "Debug_UWP" set l_build_type=Debug&set l_build_target=UWP
if /i "%p_Configuration%" == "Release" set l_build_type=Release&set l_build_target=GPU
if /i "%p_Configuration%" == "Release_CpuOnly" set l_build_type=Release&set l_build_target=CPU-only
if /i "%p_Configuration%" == "Release_UWP" set l_build_type=Release&set l_build_target=UWP
if /i "%p_Configuration%" == "Release_NoOpt" set l_build_type=Release_NoOpt&set l_build_target=GPU

echo #define _BUILDTYPE_ "%l_build_type%">> %_outfile%$$
echo #define _BUILDTARGET_ "%l_build_target%">> %_outfile%$$

:: assuming CNTK_ENABLE_ASGD was true as default value 
if "%p_CNTK_ENABLE_ASGD%" == "false" (
    echo #define _WITH_ASGD_ "no">>%_outfile%$$
) else (
    echo #define _WITH_ASGD_ "yes">>%_outfile%$$
)
if not %l_build_target% == CPU-only if not %l_build_target% == UWP (
    if "%p_CudaPath%" == "" (
        echo #define _CUDA_PATH_    "NOT_DEFINED"     >> %_outfile%$$
    ) else (
        echo #define _CUDA_PATH_    "!p_CudaPath:\=\\!" >> %_outfile%$$
    )

    if not "%p_CUDNN_PATH%" == "" (
        echo #define _CUDNN_PATH_  "%p_CUDNN_PATH:\=\\%" >> %_outfile%$$
    )

    if not "%p_CUB_PATH%" == "" (
        echo #define _CUB_PATH_  "%p_CUB_PATH:\=\\%" >> %_outfile%$$
    )
)

:: MPI info
set MPI_NAME="Unknown"
set MPI_VERSION="Unknown"
where -q mpiexec && mpiexec.exe -help > NUL 2>&1 
if not errorlevel 1 (
    for /f "tokens=1 delims= " %%i in ('mpiexec -help ^| findstr Version') do (
        if "%%i" == "Microsoft" (
            set MPI_NAME="Microsoft MPI"
            for /f "tokens=6 delims=] " %%i in ('mpiexec -help ^| findstr Version') do set MPI_VERSION="%%i"
        ) else if "%%i" == "Intel" (
            set MPI_NAME="Intel MPI"
            for /f "tokens=8 delims= " %%i in ('mpiexec -help ^| findstr Version') do set MPI_VERSION="%%i"
        )
    )
)
echo #define _MPI_NAME_ %MPI_NAME% >> %_outfile%$$
echo #define _MPI_VERSION_ %MPI_VERSION% >> %_outfile%$$

echo #endif >> %_outfile%$$

::: update file only if it changed (otherwise CNTK.cpp will get rebuilt each time)
fc %_outfile%$$ %_outfile% > NUL 2>&1
if errorlevel 1 move /Y %_outfile%$$ %_outfile%
