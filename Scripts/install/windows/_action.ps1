#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

function ActionOperations()
{
    Write-Host "Performing install operations"

    foreach ($item in $Script:operationList) {
        
        foreach ($actionItem in $item.Action) {
            ActionItem $actionItem
        }
    }

    Write-Host "Install operations finished"
    Write-Host
}

function ActionItem(
    [hashtable] $item)
{
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    Invoke-Expression $expr 
}


function InstallExe(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    
    $func = $table["Function"]
    $cmd  = $table["Command"]
    $param= $table["Param"]
    $dir  = $table["WorkDir"]
    $platform = $table["Platform"]
    $processWait = $table["ProcessWait"]
    $message =  $table["message"]
    $runAs = $table["runAs"]
    $maxErrorLevel = $table["maxErrorLevel"]

    if ($runAs -eq $null) {
        $runAs = $true
    }
    if ($maxErrorLevel -eq $null) {
        $maxErrorLevel = 0
    }
    if ($platform -ne $null) {
        $runningOn = ((Get-WmiObject -class Win32_OperatingSystem).Caption).ToUpper()
        $platform  = ($platform.ToString()).ToUpper()

        if (-not $runningOn.StartsWith($platform)) {
            return
        }
    }

    if ($message -ne $null) {
        Write-Host $message
    }
    
    if ($dir -eq $null) {
        DoProcess -command $cmd -param $param -requiresRunAs $runAs -maxErrorLevel $maxErrorLevel
    }
    else {
        DoProcess -command $cmd -param $param -requiresRunAs $runAs -workingDir $dir -maxErrorLevel $maxErrorLevel
    }
    
    if ( ($processWait -ne $null) -and ($Execute) -and ($false) ) {
        do {
            start-sleep 20
            $pwait = Get-Process $processWait -ErrorAction SilentlyContinue
        } while (-not ($pwait -eq $null))
    }
}

function InstallYml(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    
    $func = $table["Function"]
    $basePath  = $table["BasePath"]
    $env= $table["Env"]
    $ymlFile  = $table["ymlFile"]

    $envsDir = Join-Path $basePath envs
    $targetDir = Join-Path $envsDir $env

    if (test-path -path $targetDir -PathType Container) {
        $newTable = @{ Function = "InstallExe"; Command = "$basepath\Scripts\conda.exe"; Param = "env update --file `"$ymlFile`" --name `"$targetDir`""; WorkDir = "$basePath\Scripts"; runAs=$false }
    }
    else {
        $newTable = @{ Function = "InstallExe"; Command = "$basepath\Scripts\conda.exe"; Param = "env create --file `"$ymlFile`" --prefix `"$targetDir`""; WorkDir = "$basePath\Scripts"; runAs=$false }
    }

    InstallExe $newTable
}

function ExecuteApplication(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    
    $func = $table["Function"]
    $appName = $table["AppName"]
    $param= $table["Param"]
    $appDir = $table["AppDir"]
    $usePath = $table["UseEnvPath"]
    $dir  = $table["WorkDir"]
    $maxErrorLevel = $table["maxErrorLevel"]

    if ($appDir -eq $null) {
        $appDir = ""
    }
    if ($usePath -eq $null) {
        $usePath = $false
    }
    if ($maxErrorLevel -eq $null) {
        $maxErrorLevel = 0
    }

    if ($Execute) {
        $application = ResolveApplicationName -name $appName -directory $appDir -usePath $usePath
        if (-not $application) {
            throw "ExecuteApplication: Couldn't resolve program [$appName] with location directory [$appDir] and usePath [$usePath]"
        }

        if ($dir -eq $null) {
            DoProcess -command $application -param $param -maxErrorLevel $maxErrorLevel
        }
        else {
            DoProcess -command $application -param $param -workingDir $dir -maxErrorLevel $maxErrorLevel
        }
    }
}

function InstallWheel(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $BasePath     = $table["BasePath"]
    $EnvName      = $table["EnvName"]
    $message      = $table["message"]
    $whlDirectory = $table["WheelDirectory"]
    $pyVersion = $table["PyVersion"]

    Write-Host $message
    if (-not $Execute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return 
    }

    $whlFile = Get-ChildItem $cntkRootDir\cntk\Python\cntk*cp$pyVersion-cp$pyVersion*.whl
    if (-not $whlFile) {
        throw "No WHL file found at $cntkRootDir\cntk\Python"
    }
    if ($whlFile.Count -gt 1) {
        Throw "Multiple WHL files found in $cntkRootDir\cntk\Python. Please make sure it contains only the WHL file matching your CNTK download"
    }
    $whl = $whlFile.FullName

    $condaExe = Join-Path $BasePath 'Scripts\conda.exe'
    $newPaths = Invoke-DosCommand $condaExe (Write-Output ..activate cmd.exe $EnvName)  -maxErrorLevel 0

    $oldPath = $env:PATH
    $env:PATH = $newPaths + ';' + $env:PATH

    Invoke-DosCommand pip (Write-Output install $whl) -maxErrorLevel 0
    $env:PATH = $oldPath 
    return
}

function MakeDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    
    $func = $table["Function"]
    $path = $table["Path"]

    if (-not (test-path -path $path)) {
        if ($Execute) {
            New-Item $path -type directory
        }
    }
}

function AddToPath(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func    = $table["Function"]
    $dir     = $table["Dir"]
    $atStart = $table["AtStart"]
    $env     = $table["env"]

    if ($env.Length -eq 0) {
        $env = "PATH"
    }

    $pathValue = [environment]::GetEnvironmentVariable($env, "Machine")
    if ($pathValue -eq $null) {
        $pathValue = ""
    }
    $pv = $pathValue.ToLower()
    $ap = $dir.ToLower()

    if ($pv.Contains("$ap")) {
        Write-Verbose "AddToPath - path information already up-to-date" 
    }

    Write-Host Adding [$dir] to environment [$env]
    if ($atStart) {
        $pathvalue = "$dir;$pathvalue"
    }
    else {
        $pathvalue = "$pathvalue;$dir"
    }
    if ($Execute) {
        SetEnvVar -name $env -content "$pathvalue" 
    }
}

function ExtractAllFromZip(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func    = $table["Function"]
    $zipFileName = $table["zipFileName"]
    $destinationFolder = $table["destinationFolder"]

    if (-not (test-path -path $destinationFolder)) {
        throw "$destinationFolder doesn't exist"
    }
    if (-not (test-path $zipFileName -PathType Leaf)) {
        throw "$zipFileName doesn't exist"
    }

    if ($Execute) {
        $obj = new-object -com shell.application
        $zipFile = $obj.NameSpace($zipFileName)
        $destination = $obj.NameSpace($destinationFolder)

        $destination.CopyHere($zipFile.Items())
    }
}

function CreateBatch(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $filename = $table["Filename"]
    $pyVersion = $table["PyVersion"]

    if (-not $Execute) {
        Write-Host "Create-Batch [$filename]:No-Execute flag. No file created"
        return
    }

    Remove-Item -Path $filename -ErrorAction SilentlyContinue | Out-Null

    $batchScript = @"
@echo off
if /I "%CMDCMDLINE%" neq ""%COMSPEC%" " (
    echo.
    echo Please execute this script from inside a regular Windows command prompt.
    echo.
    exit /b 0
)
set PATH=$cntkRootDir\cntk;%PATH%
"$AnacondaBasePath\Scripts\activate" "$AnacondaBasePath\envs\cntk-py$pyVersion"
"@

    add-content -Path $filename -Encoding Ascii -Value $batchScript
}


function DoProcess(
    [string]  $command,
    [string]  $param,
    [string]  $workingDir = "",
    [boolean] $requiresRunAs = $false,
    [int] $maxErrorLevel)
{
    $info = "start-process [$command] with [$param]"

    Write-Verbose "$info"

    if (-not $Execute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return
    }

    if ($workingDir.Length -eq 0) {
        if ($requiresRunAs) {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru -Verb runas
        }
        else {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru
        }

    }
    else {
        if ($requiresRunAs) {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru -Verb runas -WorkingDirectory "$workingDir"
        }
        else {
            $process = start-process -FilePath "$command" -ArgumentList "$param" -Wait -PassThru -WorkingDirectory "$workingDir"
        }
    }

    $eCode = ($process.ExitCode)

    if ($ecode -gt $maxErrorLevel) {
        throw "Running 'start-process $commandString $param' failed with exit code [$ecode]"
    }
    
    return
}



function SetEnvVar(
    [Parameter(Mandatory=$true)][string] $name,
    [Parameter(Mandatory=$true)][string] $content,
    [string] $location = "Machine")
{
    Write-Verbose "SetEnvVar [$name] with [$content]"
    
    if ($Execute) {
        $commandString = "& { [environment]::SetEnvironmentVariable('"+$name+"', '"+$content+"', '"+$location+"') }"
        RunPowershellCommand -command "$commandString" -elevated $true -maxErrorLevel 0
    }    
}

function RunPowershellCommand(
    [string] $commandString,
    [boolean] $elevated,
    [int] $maxErrorLevel
)
{
    $commandBytes = [System.Text.Encoding]::Unicode.GetBytes($commandString)
    $encodedCommand = [Convert]::ToBase64String($commandBytes)
    $commandLine = "-NoProfile -EncodedCommand $encodedCommand"

    if ($elevated) {
        $process = Start-Process -PassThru -FilePath powershell.exe -ArgumentList $commandLine -wait -verb runas
    }
    else {
        $process = Start-Process -PassThru -FilePath powershell.exe -ArgumentList $commandLine -wait
    }
    
    $eCode = ($process.ExitCode)
    if ($ecode -gt $maxErrorLevel) {
        throw "Running 'powershell.exe $commandString' failed with exit code [$ecode]"
    }
    return
}

function Invoke-DosCommand {
  [CmdletBinding()]
  Param(
    [ValidateScript({ Get-Command $_ })]
    [string] $Command,
    [string[]] $Argument,
    [string] [ValidateScript({ Test-Path -PathType Container $_ })] $WorkingDirectory,
    [int] $maxErrorLevel,
    [switch] $SuppressOutput
  )
    Write-Verbose "Running '$Command $Argument'"
    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory -ErrorAction Stop
    }
    if ($SuppressOutput) {
        $null = & $Command $Argument 2>&1
    } else {
        & $Command $Argument
    }
    if ($WorkingDirectory) {
        Pop-Location
    }
    if ($LASTEXITCODE -gt $maxErrorLevel) {
        throw "Running '$Command $Argument' failed with exit code $LASTEXITCODE"
    }
}

function ResolveApplicationName(
    [Parameter(Mandatory=$True)][string] $name,
    [string] $directory = "",
    [bool] $usePath = $false)
{
    $application = ""

    if ($directory) {
        $application = CallGetCommand (join-path $directory $name)
    }
    if (-not $application) {
        if ($usePath) {
            # we are at this point if we are supposed to check in the path environment for a match and
            # $directory was empty or we couldn't find it in the $directory

            $application = CallGetCommand $name
        }
    }
    # application will be an empty string if we couldn't resolve the name, otherwise we can execute $application
    return $application
}

function CallGetCommand(
    [Parameter(Mandatory=$True)][string] $application)
{
    $matches = @(get-command $application -CommandType Application -TotalCount 1 -ErrorAction SilentlyContinue)
    if ($matches.Count -eq 0) {
        return ""
    }

    return $matches[0].Source
}

# vim:set expandtab shiftwidth=4 tabstop=4: