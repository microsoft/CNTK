@echo off 
setlocal enabledelayedexpansion

echo #ifndef _BUILDINFO_H > buildinfo.h
echo #define _BUILDINFO_H >> buildinfo.h 


FOR /F "usebackq" %%i IN (`hostname`) DO SET HOST=%%i           
:: assuming hostname always exists 

:: not sure whether git in path ? 
git --version 2 > nul 
if not %ERRORLEVEL% == 9909 (
    echo #define _GIT_EXIST >> buildinfo.h
    FOR /F "usebackq" %%i IN (`git rev-parse --abbrev-ref HEAD`) DO SET BRANCH=%%i
    FOR /F "usebackq" %%i IN (`git rev-parse HEAD`) DO SET COMMIT=%%i
    echo #define _BUILDBRANCH_  "!BRANCH!"      >> buildinfo.h
    echo #define _BUILDSHA1_    "!COMMIT!"      >> buildinfo.h
) 


echo #define _BUILDER_ "%USERNAME%"     >> buildinfo.h 
echo #define _BUILDMACHINE_ "!HOST!"    >> buildinfo.h

set a=%~dp0
set buildpath="%a:\=\\%"
echo #define _BUILDPATH_    %buildpath%     >> buildinfo.h


echo #endif >> buildinfo.h
