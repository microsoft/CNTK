@echo off
setlocal enableDelayedexpansion

::: Copyright (c) Microsoft. All rights reserved.
:::
::: Licensed under the MIT license. See LICENSE.md file in the project root 
::: for full license information.
::: ==============================================================================
:::
::: This is called as a pre-build step for the CNTK executable, taking parameters below.
::: It creates buildinfo.h, which makes version information available to the executable itself.

:: Grab the parameters
::
:: Note: don't rely on environment variables, since properties may have been
:: overridden at msbuild invocation. By convention, we let parameters start with p_, locals with l_.
:: A Vim search for [%!]\([lp]_\)\@!\w\+[%!:] should only match
:: well-known (non-CNTK-specific) environment variables.
set p_Configuration=%~1
set p_CNTK_MKL_SEQUENTIAL=%~2
set p_CNTK_ENABLE_1BitSGD=%~3
set p_CudaPath=%~4
set p_CUDNN_PATH=%~5
set p_CUB_PATH=%~6
set p_CNTK_ENABLE_ASGD=%~7

echo #ifndef _BUILDINFO_H > buildinfo.h$$
echo #define _BUILDINFO_H >> buildinfo.h$$

FOR /F %%i IN ('hostname') DO SET HOST=%%i
:: assuming hostname always exists

:: note: we'll only use git which is in path
where -q git
if not errorlevel 1 (
    call git --version > NUL 2>&1
    if not errorlevel 1 (
        echo #define _GIT_EXIST >> buildinfo.h$$
        FOR /F %%i IN ('call git rev-parse --abbrev-ref HEAD') DO SET l_BRANCH=%%i
        FOR /F %%i IN ('call git rev-parse HEAD') DO SET l_COMMIT=%%i
        set l_STATUS=
        call git diff --quiet --cached
        if not errorlevel 1 call git diff --quiet
        if errorlevel 1 set l_STATUS= ^(modified^)
        echo #define _BUILDBRANCH_  "!l_BRANCH!"      >> buildinfo.h$$
        echo #define _BUILDSHA1_    "!l_COMMIT!!l_STATUS!">> buildinfo.h$$
    )
)

if "%p_CNTK_MKL_SEQUENTIAL%" == "1" (
  echo #define _MATHLIB_ "mkl-sequential">> buildinfo.h$$
) else (
  echo #define _MATHLIB_ "mkl">> buildinfo.h$$
)

echo #define _BUILDER_ "%USERNAME%"     >> buildinfo.h$$
echo #define _BUILDMACHINE_ "%HOST%"    >> buildinfo.h$$

set l_scriptpath=%~dp0
set l_buildpath="%l_scriptpath:\=\\%"
echo #define _BUILDPATH_    %l_buildpath%     >> buildinfo.h$$

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

echo #define _BUILDTYPE_ "%l_build_type%">> buildinfo.h$$
echo #define _BUILDTARGET_ "%l_build_target%">> buildinfo.h$$

if "%p_CNTK_ENABLE_1BitSGD%" == "true" (
    echo #define _WITH_1BITSGD_ "yes">>buildinfo.h$$
) else (
    echo #define _WITH_1BITSGD_ "no">>buildinfo.h$$
)
:: assuming CNTK_ENABLE_ASGD was true as default value 
if "%p_CNTK_ENABLE_ASGD%" == "false" (
    echo #define _WITH_ASGD_ "no">>buildinfo.h$$
) else (
    echo #define _WITH_ASGD_ "yes">>buildinfo.h$$
)
if not %l_build_target% == CPU-only if not %l_build_target% == UWP (
    if "%p_CudaPath%" == "" (
        echo #define _CUDA_PATH_    "NOT_DEFINED"     >> buildinfo.h$$
    ) else (
        echo #define _CUDA_PATH_    "!p_CudaPath:\=\\!" >> buildinfo.h$$
    )

    if not "%p_CUDNN_PATH%" == "" (
        echo #define _CUDNN_PATH_  "%p_CUDNN_PATH:\=\\%" >> buildinfo.h$$
    )

    if not "%p_CUB_PATH%" == "" (
        echo #define _CUB_PATH_  "%p_CUB_PATH:\=\\%" >> buildinfo.h$$
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
echo #define _MPI_NAME_ %MPI_NAME% >> buildinfo.h$$
echo #define _MPI_VERSION_ %MPI_VERSION% >> buildinfo.h$$

echo #endif >> buildinfo.h$$

::: update file only if it changed (otherwise CNTK.cpp will get rebuilt each time)
fc buildinfo.h$$ buildinfo.h > NUL 2>&1
if errorlevel 1 move /Y buildinfo.h$$ buildinfo.h
