#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

<#
  .SYNOPSIS
 Use this cmdlet to install CNTK from a precompiled binary drop (see https://github.com/Microsoft/CNTK/releases)

 By default the script will:

 - Create or reuse Anaconda3 in the folder `C:\local\Anaconda3-4.1.1-Windows-x86_64`
 - Create or update a CNTK Python 3.5 environment in `C:\local\Anaconda3-4.1.1-Windows-x86_64\envs\cntk-py35`

 .DESCRIPTION
 The script will download and install the CNTK prerequisites and Anaconda environment.

 It will analyse your machine and will determine which components are required. 
 The required components will be downloaded and cached.
 Repeated operation of this script will reuse already downloaded components.

 - If required VS2015 Runtime will be installed
 - If required MSMPI will be installed
 - Anaconda3 will be installed into [<AnacondaBasePath>]
 - A CNTK-PY<version> environment will be created or updated in [<AnacondaBasePath>\envs]
 - CNTK will be installed or updated in the CNTK-PY<version> environment
 
 .PARAMETER Execute
 You need to supply this optional parameter to have the install script perform any changes to your machine. 
 Without this parameter NO CHANGES will be done to your machine.

 .PARAMETER AnacondaBasePath
 This optional parameter allows you to specify the location of an Anaconda installation to be used or created on your 
 machine. If the directory exists on your machine, the script will continue under the assumption that this is a working 
 Anaconda 3 (4.1.1) (or compatible) installation, and will create the CNTK Python environment in that location.
 By default a version of Anaconda3 will be installed into [C:\local\Anaconda3-4.1.1-Windows-x86_64]

 .PARAMETER PyVersion
 This is an optional parameter and can be used to specify the Python version used in the CNTK Python environment.
 Supported values for this parameter are 27, 34, or 35. The default values is 35 (for a CNTK Python 35 environment).

.EXAMPLE
 .\install.ps1
 
 Run the installer and see what operations would be performed
.EXAMPLE
 .\install.ps1 -Execute
 
 Run the installer and perform the installation operations
.EXAMPLE
 .\install.ps1 -Execute -AnacondaBasePath d:\cntkBeta

 This will install Anaconda in the [d:\cntkBeta] directory.
#>

[CmdletBinding()]
Param(
    [parameter(Mandatory=$false)] [string] $AnacondaBasePath = "C:\local\Anaconda3-4.1.1-Windows-x86_64",
    [parameter(Mandatory=$false)] [ValidateSet("27", "34", "35")] [string] $PyVersion = "35",
    [parameter(Mandatory=$false)] [switch] $Execute)

$MyDir = Split-Path $MyInvocation.MyCommand.Definition

$cntkRootDir = split-path $MyDir | split-path | Split-Path

$roboCopyCmd    = "C:\Windows\System32\robocopy.exe"
$localCache     = "$MyDir\InstallCache"

# Get the current script's directory and Dot-source the a file with common Powershell script function 
# residing in the the current script's directory
. "$MyDir\_operations"
. "$MyDir\_verify"
. "$MyDir\_download"
. "$MyDir\_info"
. "$MyDir\_action"

Function main
{
    try {
        if (-not (DisplayStart)) {
            Write-Host 
            Write-Host " ... Quitting ... "
            Write-Host
            return
        }

        if(-not (Test-Path -Path $localCache)) {
            new-item -Path $localcache -ItemType Container | Out-Null
        }

        $Script:operationList  = @()
        if (VerifyOperations) {

            DownloadOperations

            ActionOperations

            DisplayEnd
        }
    }
    catch {
        Write-Host `nFatal error during script execution!`n($Error[0]).Exception`n
    }
}

main

exit 0
