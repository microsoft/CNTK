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
 You can set this switch to 'false' to prevent Install from performing any physical changes to the machine.

 .PARAMETER NoConfirm
 If you supply this optional parameter, the install script will execute operations without asking for user confirmation.

 .PARAMETER AnacondaBasePath
 This optional parameter allows you to specify the location of an Anaconda installation to be used or created on your 
 machine. If the directory exists on your machine, the script will continue under the assumption that this is a working 
 Anaconda 3 (4.1.1) (or compatible) installation, and will create the CNTK Python environment in that location.
 By default a version of Anaconda3 will be installed into [C:\local\Anaconda3-4.1.1-Windows-x86_64]

 .PARAMETER PyVersion
 This is an optional parameter and can be used to specify the Python version used in the CNTK Python environment.
 Supported values for this parameter are 27, 34, 35, or 36. The default values is 35 (for a CNTK Python 35 environment).

  .PARAMETER WheelBaseUrl
 This is an internal test-only parameter and should be ignored.

.EXAMPLE
 .\install.ps1
 
 Run the installer and perform the installation operations
.EXAMPLE
 .\install.ps1 -Execute:$false
 
 Run the installer and see what operations would be performed, without actually performing these actions
.EXAMPLE
 .\install.ps1 -Execute -AnacondaBasePath d:\cntkBeta

 This will install Anaconda in the [d:\cntkBeta] directory.
#>

[CmdletBinding()]
Param(
    [Parameter(Mandatory=$false)] [string] $AnacondaBasePath = "C:\local\Anaconda3-4.1.1-Windows-x86_64",
    [Parameter(Mandatory=$false)] [ValidateSet("27", "34", "35", "36")] [string] $PyVersion = "35",
    [Parameter(Mandatory=$false)] [switch] $Execute = $true,
    [Parameter(Mandatory=$false)] [switch] $NoConfirm,
    [Parameter(Mandatory=$false)] [string] $WheelBaseUrl = "https://cntk.ai/PythonWheel")

Set-StrictMode -Version latest

Import-Module Download -ErrorAction Stop

$MyDir = Split-Path $MyInvocation.MyCommand.Definition
$ymlDir = Split-Path $MyDir
$cntkRootDir = Split-Path $MyDir | Split-Path | Split-Path | Split-Path

$roboCopyCmd    = "C:\Windows\System32\robocopy.exe"
$localCache     = "$MyDir\InstallCache"

. "$MyDir\_operations"
. "$MyDir\_verify"
. "$MyDir\_download"
. "$MyDir\_info"
. "$MyDir\_action"

function VerifyInstallationContent(
    [Parameter(Mandatory=$true)][string] $path)
{
    $structureCorrect = (join-path $path cntk\cntk.exe | test-path -PathType Leaf) 
    $structureCorrect = (join-path $path prerequisites\VS2015\vc_redist.x64.exe | test-path -PathType Leaf) -and $structureCorrect
    $structureCorrect = (join-path $path version.txt | test-path -PathType Leaf) -and $structureCorrect
    
    Write-Verbose "[VerifyInstallationContent]: [$path] result [$structureCorrect]"

    if (-not $structureCorrect) {
        throw "`nFatal Error: Files from the CNTK binary download package are missing!`nThe install script must be run out of the unpacked binary CNTK package. For help see: https://github.com/Microsoft/CNTK/wiki/Setup-Windows-Binary-Script"
    }
}

function WhlFileInfoFromVersionFile(
    [Parameter(Mandatory=$true)][string] $path,
    [Parameter(Mandatory=$true)][string] $pyVersion,
    [string] $wheelBaseUrl)
{
    $versionFile = Join-Path $path version.txt

    try {
        $reader = [System.IO.File]::OpenText($versionFile)
        $cntkVersion = $reader.ReadLine()       # cntk-*-*-xxxx*-*
        $cntkConfig = $reader.ReadLine()        # Debug, Release, ...
        $cntkTarget = $reader.ReadLine()        # CPU-Only, GPU, ...

        if ((-not $cntkVersion) -or (-not $cntkConfig) -or (-not $cntkTarget) -or (-not ($cntkVersion -match "^cntk"))) {
            throw "`nFatal Error: Malformed version information in [$versionFile]."
        }
        $cntkVersion = $cntkVersion -replace "-", "."
        $cntkVersion = $cntkVersion -replace "^cntk\.", "cntk-"

        return @{ Name = "{0}-cp{1}-cp{2}m-win_amd64.whl" -f $cntkVersion, $pyVersion, $pyVersion; CntkUrl = "{0}/{1}" -f $wheelBaseUrl, $cntkTarget }
    }
    finally {
        $reader.close()
    }
}

function Get-WheelUrl(
    [Parameter(Mandatory=$true)][string] $path,
    [Parameter(Mandatory=$true)][string] $pyVersion,
    [string] $WheelBaseUrl)
{
    # if a local wheel exists in the cntk\Python directory, we will pip install this wheel
    # if the file doesn't exist we will pip install the wheel in the specified url

    $whlFileInfo = WhlFileInfoFromVersionFile -path $path -pyVersion $pyVersion -wheelBaseUrl $WheelBaseUrl

    $whlPath = Join-Path $path cntk\Python
    $whlFile = Join-Path $whlPath $whlFileInfo.Name

    if (Test-Path $whlFile) {
        return $whlFile
    }

    return "{0}/{1}" -f $whlFileInfo.CntkUrl, $whlFileInfo.Name
}

Function main
{
    try {
        if (-not (DisplayStart -NoConfirm $NoConfirm)) {
            Write-Host  
            Write-Host " ... Quitting ... "
            Write-Host
            return
        }

        #check we are running inside the unpacked distribution content
        VerifyInstallationContent $cntkRootDir

        $whlUrl = Get-WheelUrl -path $cntkRootDir -pyVersion $PyVersion -wheelBaseUrl $WheelBaseUrl

        if(-not (Test-Path -Path $localCache)) {
            new-item -Path $localcache -ItemType Container | Out-Null
        }

        $operations = Set-OperationsInfo -whlUrl $whlUrl 
        $Script:operationList  = @()
        $Script:WinProduct = $null
        if (VerifyOperations -NoConfirm $NoConfirm) {

            DownloadOperations
            
            ActionOperations

            DisplayEnd
        }
    }
    catch {
        Write-Host `nFatal error during script execution!`n($Error[0]).Exception`n
        exit 1
    }
}

main

exit 0
