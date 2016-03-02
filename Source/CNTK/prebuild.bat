@echo off
setlocal enableDelayedexpansion

::: This is called as a pre-build step for the CNTK executable.
::: It receives the build's configuration, $(Configuration), as first paramter.
::: It creates buildinfo.h, which makes version information available to the executable itself.

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
        FOR /F %%i IN ('call git rev-parse --abbrev-ref HEAD') DO SET BRANCH=%%i
        FOR /F %%i IN ('call git rev-parse HEAD') DO SET COMMIT=%%i
        set STATUS=
        call git diff --quiet --cached
        if not errorlevel 1 call git diff --quiet
        if errorlevel 1 set STATUS= ^(modified^)
        echo #define _BUILDBRANCH_  "!BRANCH!"      >> buildinfo.h$$
        echo #define _BUILDSHA1_    "!COMMIT!!STATUS!">> buildinfo.h$$
    )
)

:: For now, math lib is basically hardwired
if exist ACML_PATH (
    echo #define _MATHLIB_ "acml">> buildinfo.h$$
)

echo #define _BUILDER_ "%USERNAME%"     >> buildinfo.h$$

echo #define _BUILDER_ "%USERNAME%"     >> buildinfo.h$$
echo #define _BUILDMACHINE_ "%HOST%"    >> buildinfo.h$$

set scriptpath=%~dp0
set buildpath="%scriptpath:\=\\%"
echo #define _BUILDPATH_    %buildpath%     >> buildinfo.h$$

set build_type=Unknown
set build_target=Unknown
if /i "%~1" == "Debug" set build_type=Debug&set build_target=GPU
if /i "%~1" == "Debug_CpuOnly" set build_type=Debug&set build_target=CPU-only
if /i "%~1" == "Release" set build_type=Release&set build_target=GPU
if /i "%~1" == "Release_CpuOnly" set build_type=Release&set build_target=CPU-only

echo #define _BUILDTYPE_ "%build_type%">> buildinfo.h$$
echo #define _BUILDTARGET_ "%build_target%">> buildinfo.h$$

if "%CNTK_ENABLE_1BitSGD%" == "true" (
    echo #define _WITH_1BITSGD_ "yes">>buildinfo.h$$
) else (
    echo #define _WITH_1BITSGD_ "no">>buildinfo.h$$
)

if not %build_target% == CPU-only (
    if "%cuda_path%" == "" (
        echo #define _CUDA_PATH_    "NOT_DEFINED"     >> buildinfo.h$$
    ) else (
        echo #define _CUDA_PATH_    "%cuda_path:\=\\%" >> buildinfo.h$$
    )

    if not "%cudnn_path%" == "" (
        echo #define _CUDNN_PATH_  "%cudnn_path:\=\\%" >> buildinfo.h$$
    )

    if not "%cub_path%" == "" (
        echo #define _CUB_PATH_  "%cub_path:\=\\%" >> buildinfo.h$$
    )
)

echo #endif >> buildinfo.h$$

::: update file only if it changed (otherwise CNTK.cpp will get rebuilt each time)
fc buildinfo.h$$ buildinfo.h > NUL 2>&1
if errorlevel 1 move /Y buildinfo.h$$ buildinfo.h
