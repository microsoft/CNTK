#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function FunctionIntro(
    [Parameter(Mandatory = $true)][hashtable] $table) {
    $table | Out-String | Write-Verbose
}

function GetKey(
    [string] $validChar) {
    do {
        $key = Read-Host
    } until ($key -match $validChar)

    return $key
}

function DisplayStartMessage {
"

This script will setup CNTK, the CNTK prerequisites and the CNTK Python environment onto the system.
More help can be found at: 
  https://github.com/Microsoft/CNTK/wiki/Setup-Windows-Binary-Script

The script will analyse your machine and will determine which components are required. 
The required components will be downloaded in [$localCache]
Repeated operation of this script will reuse already downloaded components.

 - If required VS2015 Runtime will be installed
 - If required MSMPI will be installed
 - Anaconda3 will be installed into [$AnacondaBasePath]
 - A CNTK-PY$PyVersion environment will be created or updated in [$AnacondaBasePath\envs]
 - CNTK will be installed or updated in the CNTK-PY$PyVersion environment
"
}

function Display64BitWarningMessage {
    "
A 64bit version of Powershell is required to run this script.
Please make sure you started this installation from a 64bit command process.
"
}

function DisplayVersionWarningMessage(
    [string] $version) {
    "
You are executing this script from Powershell Version $version.
We recommend that you execute the script from Powershell Version 4 or later. You can install Powershell Version 4 from:
    https://www.microsoft.com/en-us/download/details.aspx?id=40855
"
}

function DisplayWarningNoExecuteMessage {
    "
The parameter '-Execute:$false' has been supplied to the script.
The script will execute without making any actual changes to the machine.
"
}

function DisplayStartContinueMessage {
    "
1 - I agree and want to continue
Q - Quit the installation process
"
}

function CheckPowershellVersion {
    $psVersion = $PSVersionTable.PSVersion.Major
    if ($psVersion -ge 4) {
        return $true
    }

    Write-Host $(DisplayVersionWarningMessage $psVersion)
    if ($psVersion -eq 3) {
        return $true
    }
    return $false
}

function CheckOSVersion {
    $runningOn = (Get-WmiObject -class Win32_OperatingSystem).Caption
    $isMatching = ($runningOn -match "^Microsoft Windows (8\.1|10|Server 2012 R2|Server 2016)") 
    if ($isMatching) {
        return
    }

    Write-Warning "
You are running this script on [$runningOn].
The Microsoft Cognitive Toolkit is designed and tested on Windows 8.1, Windows 10, 
Windows Server 2012 R2, and Windows Server 2016.
"
    return
}

function Check64BitProcess {
    if ([System.Environment]::Is64BitProcess) {
        return $true
    }

    Write-Warning $(Display64BitWarningMessage)
    return $false
}

function DisplayStart(
    [bool] $NoConfirm) {
    Write-Host $(DisplayStartMessage)
    if (-not (Check64BitProcess)) {
        return $false
    }
    if (-not (CheckPowershellVersion)) {
        return $false
    }

    CheckOSVersion

    if (-not $Execute) {
        Write-Warning $(DisplayWarningNoExecuteMessage)
    }
    if ($NoConfirm) {
        return $true
    }
    Write-Host $(DisplayStartContinueMessage)
    $choice = GetKey '^[1qQ]+$'

    if ($choice -contains "1") {
        return $true
    }

    return $false
}

Function DisplayEnd() {
    if (-not $Execute) { return }

    Write-Host "

CNTK v2 Python install complete.

To activate the CNTK Python environment and set the PATH to include CNTK, start a command shell and run
   $cntkRootDir\scripts\cntkpy$PyVersion.bat

Please checkout tutorials and examples here:
   $cntkRootDir\Tutorials
   $cntkRootDir\Examples

"
}
