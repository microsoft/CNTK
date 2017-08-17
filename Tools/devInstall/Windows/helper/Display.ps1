#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function FunctionIntro(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    $table | Out-String | Write-Verbose
}

function GetKey(
    [string] $validChar
)
{
    do {
        $key = Read-Host
    } until ($key -match $validChar)

    return $key
}

function DisplayStartMessage
{
"

This script will setup the CNTK Development Environment on your machine.
More help is given by calling get-help .\devInstall.ps1

The script will analyse your machine and will determine which components are required. 
The required components will be downloaded into [$localCache]
Repeated operation of this script will reuse already downloaded components.
"
}

function DisplayVersionWarningMessage(
    [string] $version)
{
"
You are executing this script from Powershell Version $version.
We recommend that you execute the script from Powershell Version 4 or later. You can install Powershell Version 4 from:
    https://www.microsoft.com/en-us/download/details.aspx?id=40855
"
}

function Display64BitWarningMessage
{
"
A 64bit version of Powershell is required to run this script.
Please check the short-cut/command to start Powershell and make sure you start the 64bit version of Powershell.
"
}

function DisplayWarningNoExecuteMessage
{
"
The parameter '-Execute' hasn't be supplied to the script.
The script will execute withouth making any actual changes to the machine!
"
}

function DisplayStartContinueMessage
{
"
1 - I agree and want to continue
Q - Quit the installation process
"
}

function CheckPowershellVersion
{
    $psVersion = $PSVersionTable.PSVersion.Major
    if ($psVersion -ge 4) {
        return $true
    }

    Write-Warning $(DisplayVersionWarningMessage $psVersion)
    if ($psVersion -eq 3) {
        return $true
    }
    return $false
}

function Check64BitProcess
{
    if ([System.Environment]::Is64BitProcess) {
        return $true
    }

    Write-Warning $(Display64BitWarningMessage)

    return $false
}

function CheckOSVersion 
{
    $runningOn = (Get-WmiObject -class Win32_OperatingSystem).Caption
    $isMatching = ($runningOn -match "^Microsoft Windows (8\.1|10|Server 2012 R2|Server 2016)") 

    if (-not $isMatching) {
       Write-Warning "
You are running this script on [$runningOn].
The Microsoft Cognitive Toolkit is designed and tested on Windows 8.1, Windows 10, 
Windows Server 2012 R2, and Windows Server 2016. 
"
    }
}

function DisplayStart(
    [bool] $NoConfirm)
{
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


Function DisplayEnd() 
{
    Write-Host "

Installation finished.
"
}

function DisplayAfterVerify(
    [bool] $NoConfirm,
    [array] $list = @())
{
    Write-Host 

    if ($list.Count -gt 0) {
        Write-Host "The following operations will be performed:"

        foreach ($item in $list) {
            $info = $item.ActionInfo
            Write-Host " * $info"
        }
        if (-not $Execute) {
           Write-Warning $(DisplayWarningNoExecuteMessage)
        }
    
        if ($NoConfirm) {
            return $true
        }

        Write-Host 
        Write-Host "Do you want to continue? (y/n)"
        
        $choice = GetKey '^[yYnN]+$'

        if ($choice -contains "y") {
            return $true
        }
    }
    else {
        Write-Host "No additional installation required"
        Write-Host
       
    }
    return $false
}

# vim:set expandtab shiftwidth=4 tabstop=4: