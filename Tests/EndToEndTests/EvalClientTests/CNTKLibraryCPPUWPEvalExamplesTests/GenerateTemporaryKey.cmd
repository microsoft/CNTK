@echo off

set ISSUER_NAME=CNTK Development
set OUTPUT_NAME=CNTKDevelopmentTemporaryKey
set OUTPUT_DIR=%~dp0\GeneratedCode

echo.
echo **********************************************************************************************
echo This script invokes makecert, which displays a modal dialog when creating a certificate. Leave 
echo this value blank (click "None") as we will be deleting the private key and creating a pfx file 
echo where the key cannot be exported.
echo **********************************************************************************************
echo.

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

makecert -sv "%OUTPUT_DIR%\%OUTPUT_NAME%.pvk" -n "CN=%ISSUER_NAME%" -r "%OUTPUT_DIR%\%OUTPUT_NAME%.cer" -eku 1.3.6.1.5.5.7.3.3 -a sha256 -sky signature -cy end -ss CA
if exist "%OUTPUT_DIR%\%OUTPUT_NAME%.pfx" del "%OUTPUT_DIR%\%OUTPUT_NAME%.pfx"
pvk2pfx -pvk "%OUTPUT_DIR%\%OUTPUT_NAME%.pvk" -spc "%OUTPUT_DIR%\%OUTPUT_NAME%.cer" -pfx "%OUTPUT_DIR%\%OUTPUT_NAME%.pfx"

REM We don't want to be able to use this certificate with other apps, so delete the pvk and cer files.
REM     - The pvk is the private key file
REM     - The cer files is the certificate file that is embedded in the pfx file.
del "%OUTPUT_DIR%\%OUTPUT_NAME%.pvk"
del "%OUTPUT_DIR%\%OUTPUT_NAME%.cer"
