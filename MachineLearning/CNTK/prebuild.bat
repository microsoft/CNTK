@echo off 
setlocal enabledelayedexpansion

::: This is called as a pre-build step for the CNTK executable.
::: It creates buildinfo.h, which makes version information available to the executable itself.

echo #ifndef _BUILDINFO_H > buildinfo.h$$
echo #define _BUILDINFO_H >> buildinfo.h$$ 

FOR /F "usebackq" %%i IN (`hostname`) DO SET HOST=%%i           
:: assuming hostname always exists 

:: not sure whether git in path ? 
git --version 2 > nul 
if not %ERRORLEVEL% == 9009 (
    echo #define _GIT_EXIST >> buildinfo.h$$
    FOR /F "usebackq" %%i IN (`git rev-parse --abbrev-ref HEAD`) DO SET BRANCH=%%i
    FOR /F "usebackq" %%i IN (`git rev-parse HEAD`) DO SET COMMIT=%%i
    echo #define _BUILDBRANCH_  "!BRANCH!"      >> buildinfo.h$$
    echo #define _BUILDSHA1_    "!COMMIT!"      >> buildinfo.h$$
)

echo #define _BUILDER_ "%USERNAME%"     >> buildinfo.h$$ 
echo #define _BUILDMACHINE_ "!HOST!"    >> buildinfo.h$$

set a=%~dp0
set buildpath="%a:\=\\%"
echo #define _BUILDPATH_    %buildpath%     >> buildinfo.h$$

set cuda_path="%CUDA_PATH:\=\\%"
echo #define _CUDA_PATH_    %cuda_path%     >> buildinfo.h$$

echo #endif >> buildinfo.h$$

::: update file only if it changed (otherwise CNTK.cpp will get rebuilt each time)
fc buildinfo.h$$ buildinfo.h > NUL 2>&1
if ERRORLEVEL 1 move /Y buildinfo.h$$ buildinfo.h
