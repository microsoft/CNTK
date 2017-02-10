#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
<#
  .SYNOPSIS
 Use this cmdlet to install a CNTK development environment on your machine.
 A detailed description can be found here: https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-with-script-on-Windows
 
 .DESCRIPTION
 The script will download and install the files necessary to create a CNTK development environment on your system. 

 It will analyse your machine and will determine which components are required. 
 The required components will be downloaded into [c:\installCacheCntk] and installed from that location.
 Repeated operation of this script will reuse already downloaded components.
 
 Before you can run this machine you should have read the instructions at 
     https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-with-script-on-Windows
 
 .PARAMETER Execute
 You need to supply this optional parameter to have the install script perform any changes to your machine. 
 Without this parameter NO CHANGES will be done to your machine.

 .PARAMETER NoConfirmation
 If you supply this optional parameter, the install script will perform operations without asking for user confirmation.
 
 .PARAMETER localCache
 This optional parameter can be used to specify the directory downloaded components will be stored in

 .PARAMETER AnacondaBasePath
 This optional parameter allows you to specify the location of an Anaconda installation to be used or created on your 
 machine. If the directory exists on your machine, the script will continue under the assumption that this is a working 
 Anaconda 3 (4.1.1) (or compatible) installation, and will create the CNTK Python environment in that location.
 By default a version of Anaconda3 will be installed into [C:\local\Anaconda3-4.1.1-Windows-x86_64]

  .PARAMETER PyVersion
 This is an optional parameter and can be used to specify the Python version used in the CNTK Python environment.
 Supported values for this parameter are 27, 34, or 35. The default values is 35 (for a CNTK Python 35 environment).

  .PARAMETER PyEnvironmentName
 This optional parameter allows to specify the name of the environment that will be created during the installation process.
 By default the environment will be named cntkdev-py<PyVersion>, where PyVersion is being replaced by the content of the <PyVersion>
 parameter to this script. If this parameter is specified by you, no version substitution in the environment will be performed. 

  .PARAMETER NoPythonEnvironment
 If this switch parameter is set, the install script will not create a CNTK Python environment during the installation process.
 This allows creation of the desired environment after the installation.

 .EXAMPLE
 .\devInstall.ps1
 
 Run the installer and see what operations would be performed
 .EXAMPLE
 .\devInstall.ps1 -Execute
 
 Run the installer and install the development tools
 
 .EXAMPLE
 .\devInstall.ps1 -Execute -AnacondaBasePath d:\mytools\Anaconda34

 If the directory [d:\mytools\Anaconda34] exists, the installer will assume it contains a complete Anaconda installation. 
 If the directory doesn't exist, Anaconda will be installed into this directory.

#>

[CmdletBinding()]
Param(
    [parameter(Mandatory=$false)] [switch] $Execute,
    [parameter(Mandatory=$false)] [string] $localCache = "c:\installCacheCntk",
    [parameter(Mandatory=$false)] [string] $InstallLocation = "c:\local",
    [parameter(Mandatory=$false)] [string] $AnacondaBasePath = "C:\local\Anaconda3-4.1.1-Windows-x86_64",
    [parameter(Mandatory=$false)] [switch] $NoConfirmation,
    [parameter(Mandatory=$false, ParameterSetName = "PythonVersion")] [ValidateSet("27", "34", "35")] [string] $PyVersion = "35",
    [parameter(Mandatory=$false, ParameterSetName = "PythonVersion")] [string] $PyEnvironmentName = "",
    [parameter(Mandatory=$true, ParameterSetName = "PythonNoEnvironment")] [switch] $NoPythonEnvironment)
    
$roboCopyCmd = "robocopy.exe"

#just make sure the supplied parameter don't end on a backslash
$localCache = (Join-Path $localCache .) | Split-Path
$InstallLocation = (Join-Path $InstallLocation .) | Split-Path
$AnacondaBasePath = (Join-Path $AnacondaBasePath .) | Split-Path

$localDir = $InstallLocation


# Get the current script's directory
$MyDir = Split-Path $MyInvocation.MyCommand.Definition

$CloneDirectory = Split-Path $mydir
$CloneDirectory = Split-Path $CloneDirectory
$CloneDirectory = Split-Path $CloneDirectory

$reponame = Split-Path $CloneDirectory -Leaf
$repositoryRootDir = Split-Path $CloneDirectory

$solutionfile = Join-Path $CloneDirectory "CNTK.SLN"

if (-not (Test-Path -Path $solutionFile -PathType Leaf)) {
    Write-Warning "The install script was started out of the [$mydir] location. Based on this"
    Write-Warning "[$CloneDirectory] should be the location of the CNTK sourcecode directory."
    Write-Warning "The specified directory is not a valid clone of the CNTK Github project."
    throw "Terminating install operation"
}


. "$MyDir\helper\Display"
. "$MyDir\helper\Common"
. "$MyDir\helper\Operations"
. "$MyDir\helper\Verification"
. "$MyDir\helper\Download"
. "$MyDir\helper\Action"
. "$MyDir\helper\PreRequisites"

Function main
{
    try { if (-not (DisplayStart -NoConfirmation $NoConfirmation)) {
            Write-Host 
            Write-Host " ... Quitting ... "
            Write-Host
            return
        }
    
        if (-not (Test-Path -Path $localCache)) {
            new-item -Path $localcache -ItemType Container -ErrorAction Stop | Out-Null
        }

        ClearScriptVariables

        $operation = @();
        $operation += OpScanProgram
        $operation += OpCheckVS15Update3

        $operation += OpCheckCuda8
        $operation += OpNVidiaCudnn5180 -cache $localCache -targetFolder $localDir
        $operation += OpNvidiaCub141 -cache $localCache -targetFolder $localDir

        $operation += OpCMake362 -cache $localCache
        $operation += OpMSMPI70 -cache $localCache
        $operation += OpMSMPI70SDK -cache $localCache
        $operation += OpBoost160VS15 -cache $localCache -targetFolder $localDir
        $operation += OpCNTKMKL3 -cache $localCache -targetFolder $localDir
        $operation += OpSwig3010 -cache $localCache -targetFolder $localDir
        $operation += OpProtoBuf310VS15 -cache $localCache -targetFolder $localDir -repoDirectory $CloneDirectory
        $operation += OpProtoBuf310VS15Prebuild -cache $localCache -targetFolder $localDir
        $operation += OpZlibVS15 -cache $localCache -targetFolder $localDir -repoDirectory $CloneDirectory
        $operation += OpZlibVS15Prebuild -cache $localCache -targetFolder $localDir
        $operation += OpOpenCV31 -cache $localCache -targetFolder $localDir
        $operation += OpAnaconda3411 -cache $localCache -AnacondaBasePath $AnacondaBasePath
        if (-not $NoPythonEnvironment) {
            $operation += OpAnacondaEnv -AnacondaBasePath $AnacondaBasePath -repoDir $repositoryRootDir -repoName $reponame -environmentName $PyEnvironmentName -pyVersion $PyVersion
        }

        $operationList = @()
        $operationList += (VerifyOperations $operation)

        PreReqOperations $operationList

        if (DisplayAfterVerify -NoConfirmation $NoConfirmation -list $operationList) {

            DownloadOperations $operationList

            ActionOperations $operationList 

            DisplayEnd
        }
    }
    catch {
        Write-Host "Exception caught - function main / failure"
        Write-Host ($Error[0]).Exception
        Write-Host
    }
}

main
exit 0

# vim:set expandtab shiftwidth=4 tabstop=4: