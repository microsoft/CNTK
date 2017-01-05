<#
  .SYNOPSIS
 Use this cmdlet to install a CNTK development environment on your machine

 .DESCRIPTION
 The script will download and install the files necessary to create a CNTK development environment on your system. 

 It will analyse your machine and will determine which components are required. 
 The required components will be downloaded in [c:\installCacheCntk]
 Repeated operation of this script will reuse already downloaded components.

 .PARAMETER Execute
 This is an optional parameter. Without setting this switch, no changes to the machine setup/installation will be performed

  .PARAMETER NoGpu
 This is an optional parameter. By setting this switch the GPU specific tools (Cuda, CuDnn, Cub) will not be installed.

 .PARAMETER localCache
 This optional parameter can be used to specify the directory downloaded components will be stored in

 .PARAMETER ServerLocation
 This is an optional parameter. The script can install pre-compiled components, this parameter 
 specifies the location on a server where this componentents are downloaded from.
 This is useful for a team environment to share the components which need to get compiled (Protobuf, Zlib, libzip) 
 
 .PARAMETER CloneDirectory
 By default the installer should be executed out of the <CntkCloneRoot>\Scripts\devInstall\Windows directory. Out of this 
 location the installer computes the root directory of the CNTK clone. In the case the installer is in a different location,
 the root directory of the CNTK clone can be specified using this parameter
 
.EXAMPLE
 installer.ps1
 
 Run the installer and see what operations would be performed
.EXAMPLE
 installer.ps1 -Execute
 
 Run the installer and install the development tools
.EXAMPLE
 installer.ps1 -Execute -NoGpu
 
 Run the installer, but don't install any GPU specific tools
.EXAMPLE
 installer.ps1 -Execute -NoGpu -CloneDirectory c:\repos\CNTKAlternate
 
 Run the installer, but don't install any GPU specific tools
.EXAMPLE

#>

[CmdletBinding()]
Param(
    [parameter(Mandatory=$false)] [switch] $Execute,
    [parameter(Mandatory=$false)] [switch] $NoGpu,
    [parameter(Mandatory=$false)] [string] $localCache = "c:\installCacheCntk",
    [parameter(Mandatory=$false)] [string] $InstallLocation = "c:\local",
    [parameter(Mandatory=$false)] [string] $ServerLocation,
    [parameter(Mandatory=$false)] [string] $CloneDirectory)
    


    $Execute = $true


$roboCopyCmd = "robocopy.exe"
$localDir = $InstallLocation


# Get the current script's directory and Dot-source the a file with common Powershell script function residing in the the current script's directory
$MyDir = Split-Path $MyInvocation.MyCommand.Definition

if ($CloneDirectory) {
    $reponame = Split-Path $CloneDirectory -Leaf
    $repositoryRootDir = Split-Path $CloneDirectory

    $solutionfile = join-path $clontDirectory "CNTK.SLN"

    if (-not (Test-Path -Path $solutionFile -PathType Leaf)) {
        Write-Warning "[$CloneDirectory] was specified as the location of the CNTK sourcecode directory."
        Write-Warning "The specified directory is not a valid clone of the CNTK Github project."
        throw "Terminating install operation"
    }
}
else {
    $CloneDirectory = Split-Path $mydir
    $CloneDirectory = Split-Path $CloneDirectory
    $CloneDirectory = Split-Path $CloneDirectory

    $reponame = Split-Path $CloneDirectory -Leaf
    $repositoryRootDir = Split-Path $CloneDirectory

    $solutionfile = join-path $CloneDirectory "CNTK.SLN"

    if (-not (Test-Path -Path $solutionFile -PathType Leaf)) {
        Write-Warning "The install script was started out of the [$mydir] location. Based on this"
        Write-Warning "[$CloneDirectory] should be the location of the CNTK sourcecode directory."
        Write-Warning "The specified directory is not a valid clone of the CNTK Github project."
        throw "Terminating install operation"
    }
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
    try { if (-not (DisplayStart)) {
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

        if (-not $NoGpu) {
            $operation += OpCheckCuda8

            $operation += OpNVidiaCudnn5180 -cache $localCache -targetFolder $localDir
            $operation += OpNvidiaCub141 -cache $localCache -targetFolder $localDir
        }

        $operation += OpCMake362 -cache $localCache
        $operation += OpMSMPI70 -cache $localCache
        $operation += OpMSMPI70SDK -cache $localCache
        $operation += OpBoost160VS15 -cache $localCache -targetFolder $localDir
        $operation += OpCNTKMKL3 -cache $localCache -targetFolder $localDir
        $operation += OpSwig3010 -cache $localCache -targetFolder $localDir
        $operation += OpProtoBuf310VS15 -cache $localCache -targetFolder $localDir
        $operation += OpZlibVS15 -cache $localCache -targetFolder $localDir
        if ($ServerLocation) {
            $operation += OpProtoBuf310VS15Internal -server $ServerLocation -cache $localCache -targetFolder $localDir
            $operation += OpZLibVS15Internal -server $ServerLocation -cache $localCache -targetFolder $localDir
        }
        $operation += OpAnaconda3411 -cache $localCache -targetFolder $localDir
        $operation += OpAnacondaEnv34 -targetFolder $localDir -repoDir $repositoryRootDir -repoName $reponame


        #$operation += OpGit2101 -cache $localCache
        #$operation += OpGitClone -targetFolder $repositoryRootDir -targetDir $reponame
        #$operation += OpSysinternals -cache $localCache -targetFolder $localDir
        #$operation += OpOpenCVInternal $ServerLocation -cache $localCache -targetFolder $localDir
        #$operation += OpOpenCV31 -cache $localCache -targetFolder $localDir
        #$operation += OpCygwin -cache $localCache -targetFolder $localDir

        #$operation += AddOpDisableJITDebug
        #$operation += OpTestData "c:\Data\CNTKTestData" "\\storage.ccp.philly.selfhost.corp.microsoft.com\public\CNTKTestData"
        #$operation += OpSysinternals -cache $localCache -targetFolder $localDir


        $operationList = @()
        $operationList += (VerifyOperations $operation)

        PreReqOperations $operationList

        if (DisplayAfterVerify $operationList) {

            DownloadOperations $operationList 

            ActionOperations $operationList 

            #DisplayEnd
        }
    }
    catch {
        Write-Host "Exception caught - function main / failure"
        Write-Host ($Error[0]).Exception
    }
}

main

exit 0

# vim:set expandtab shiftwidth=2 tabstop=2: