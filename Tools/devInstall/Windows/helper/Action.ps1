#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function ActionOperations(
    [parameter(Mandatory=$true)][array] $actionList)
{
    Write-Host "Performing install operations"

    foreach ($item in $actionList) {
        if ($item.ActionInfo) {
            Write-Host $item.ActionInfo
        }
        foreach ($actionItem in $item.Action) {
            ActionItem $actionItem
        }
    }
}

function ActionItem(
    [hashtable] $item)
{
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    Invoke-Expression $expr 
}

function InstallExeForPlatform(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $platform = $table["Platform"]

    if (PlatformMatching $platform) {
        InstallExe $table
    }
}

function InstallExe(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    
    $func = $table["Function"]
    $cmd  = $table["Command"]
    $param= $table["Param"]
    $dir  = $table["WorkDir"]
    $processWait = $table["ProcessWait"]
    $message =  $table["message"]
    $runAs = GetTableDefaultBool -table $table -entryName "runAs" -defaultValue $true
    $maxErrorLevel = GetTableDefaultInt -table $table -entryName "maxErrorLevel" -defaultValue 0

    if ($message -ne $null) {
        Write-Host $message
    }
    
    if ($dir -eq $null) {
        DoProcess -doExecute $Execute -command $cmd -param $param -requiresRunAs $runAs -maxErrorLevel $maxErrorLevel -throwOnError $true
    }
    else {
        DoProcess -doExecute $Execute -command $cmd -param $param -requiresRunAs $runAs -workingDir $dir -maxErrorLevel $maxErrorLevel -throwOnError $true
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
    $condaPath = Join-Path $basePath Scripts\conda.exe

    if (test-path -path $targetDir -PathType Container) {
        $newTable = @{ Function = "InstallExe"; Command = $condaPath; Param = "env update --file `"$ymlFile`" --name $targetDir"; WorkDir = "$basePath\Scripts"; runAs=$false }
    }
    else {
        $newTable = @{ Function = "InstallExe"; Command = $condaPath; Param = "env create --file `"$ymlFile`" --prefix $targetDir"; WorkDir = "$basePath\Scripts"; runAs=$false }
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
    $dir  = $table["WorkDir"]
    $appDir = GetTableDefaultString -table $table -entryName "AppDir" -defaultValue ""
    $usePath = GetTableDefaultBool -table $table -entryName "UseEnvPath" -defaultValue $false
    $maxErrorLevel = GetTableDefaultInt -table $table -entryName "maxErrorLevel" -defaultValue 0

    if (-not $Execute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return 
    }
    $application = ResolveApplicationName -name $appName -directory $appDir -usePath $usePath
    if (-not $application) {
        throw "ExecuteApplication: Couldn't resolve program [$appName] with location directory [$appDir] and usePath [$usePath]"
    }
    if ($dir -eq $null) {
        DoProcess -doExecute $Execute -command $application -param $param -maxErrorLevel $maxErrorLevel -throwOnError $true
    }
    else {
        DoProcess -doExecute $Execute -command $application -param $param -workingDir $dir -maxErrorLevel $maxErrorLevel -throwOnError $true
    }
}

function InstallWheel(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $BasePath     = $table["BasePath"]
    $EnvName      = $table["EnvName"]
    $whl          = $table["whl"]
    $message      = $table["message"]
    $whlDirectory = $table["WheelDirectory"]

    Write-Host $message
    if (-not $Execute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return 
    }
    $condaExe = Join-Path $BasePath 'Scripts\conda.exe'
    $newPaths = Invoke-DosCommand $condaExe (Write-Output ..activate cmd.exe $EnvName) -maxErrorLevel 0

    $oldPath = $env:PATH
    $env:PATH = $newPaths + ';' + $env:PATH
    
    Invoke-DosCommand pip (Write-Output install $whl) -maxErrorLevel 0
    $env:PATH = $oldPath 
}

function InstallMSI(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
   
    $func = $table["Function"]
    $msi  = $table["MsiName"]
    $dir  = $table["MsiDir"]

    $completeMsiName = Join-Path $dir $msi

    $windDir = GetEnvironmentVariableContent("WINDIR")
    $cmd = Join-Path $windDir System32\MSIEXEC.EXE

    $param= "/i `"$completeMsiName`" /quiet /norestart"
  
    DoProcess -doExecute $Execute -command $cmd -param $param -requiresRunAs $true -maxErrorLevel 0 -throwOnError $true
}
               
function MakeDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table
    
    $func = $table["Function"]
    $path = $table["Path"]

    if (-not (test-path -path $path)) {
        if ($Execute) {
            New-Item $path -type directory | Out-Null
        }
    }
}

function RobocopySourceDestination(
    [Parameter(Mandatory = $true)][string] $source,
    [Parameter(Mandatory = $true)][string] $destination,
    [bool] $copyAdditive=$false)
{
    if (-not (test-path $source -PathType Any)) {
        throw SourceDirectory [$source] is missing
    }

    $option = "/NFL /copy:DT /dcopy:D /xj"
    if (-not $copyAdditive) {
        $option += " /MIR "
    }

    $param = "`"$source`" `"$destination`" $option"

    DoProcess -doExecute $Execute -command $roboCopyCmd -param $param -maxErrorLevel 8 -throwOnError $true
}

function SetEnvironmentVariable(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func    = $table["Function"]
    $name    = $table["EnvVar"]
    $content = $table["Content"]
    $location = "User"

    if (-not $Execute) {
        return
    }
    else {
        $demoMessage = ""
    }

    if ($demoMessage.Length -gt 0) {
        Write-Verbose "$demoMessage[$func]: [$name] = [$content] in [$location]"
    }

    SetUserEnvironmentVariable -name "$name" -content "$content" 
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
        return
    }

    Write-Verbose "Adding [$dir] to environment [$env]"
    if ($atStart) {
        $pathvalue = "$dir;$pathvalue"
    }
    else {
        $pathvalue = "$pathvalue;$dir"
    }
    if ($Execute) {
        SetUserEnvironmentVariable -name $env -content $pathvalue
    }
}

function ExtractAllFromZipForPlatform(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $platform = $table["Platform"]

    if (PlatformMatching $platform) {
        ExtractAllFromZip $table
    }
}

function ExtractAllFromZip(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $zipFileName = $table["zipFileName"]
    $destination = $table["destination"]
    $destinationFolder = $table["destinationFolder"]
    $zipSubTree = $table["zipSubTree"]
    $copyAdditive = GetTableDefaultBool -table $table -entryName "AddToDirectory" -defaultValue $false
    
    Write-Verbose "ExtractAllFromZip: zipFileName [$zipFileName] destination [$destination] Folder [$destinationFolder]"

    if (-not $Execute) {
        return
    }
    $completeDestination = join-path -Path $destination -ChildPath $destinationFolder

    if (-not (test-path -path $completeDestination)) {
        new-item $completeDestination -type directory -Force -ErrorAction Stop | Out-Null
    }
    if (-not (test-path $zipFileName -PathType Leaf)) {
        throw  "ExtractAllFromZip: zipFileName [$zipFileName] not found!"
    }

    $tempDir = [System.IO.Path]::GetTempFileName();
    remove-item $tempDir | Out-Null

    $completeTempDestination = join-path -Path $tempDir -ChildPath $destinationFolder
    new-item -type directory -path $completeTempDestination -Force -ErrorAction Stop | Out-Null

    $obj = new-object -com shell.application
    $zipFile = $obj.NameSpace($zipFileName)
    $destinationNS = $obj.NameSpace($completeTempDestination)

    $destinationNS.CopyHere($zipFile.Items())

    if ($zipSubTree) {
        $completeTempDestination = join-path $completeTempDestination $zipSubTree
    }

    RobocopySourceDestination $completeTempDestination $completeDestination $copyAdditive

    rm -r $tempDir -Force -ErrorAction SilentlyContinue | Out-Null
}


function Extract7zipSelfExtractingArchive(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $archiveName = $table["archiveName"]
    $destination = $table["destination"]
    $destinationFolder = $table["destinationFolder"]
    $archiveSubTree = $table["archiveSubTree"]
    $copyAdditive = GetTableDefaultBool -table $table -entryName "AddToDirectory" -defaultValue $false
    
    Write-Verbose "Extract7zipSelfExtractingArchive: archiveName [$archiveName] destination [$destination] Folder [$destinationFolder]"

    if (-not $Execute) {
        return
    }
    $completeDestination = join-path -Path $destination -ChildPath $destinationFolder

    if (-not (test-path $archiveName -PathType Leaf)) {
        throw  "Extract7zipSelfExtractingArchive: zipFileName [$zipFileName] not found!"
    }

    $tempDir = [System.IO.Path]::GetTempFileName();
    remove-item $tempDir | Out-Null

    $completeTempDestination = $tempDir
    new-item -type directory -path $completeTempDestination -Force -ErrorAction Stop | Out-Null

    $cmdParm  = "-o`"$completeTempDestination`" -y"
    $newTable = @{ Function = "InstallExe"; Command = $archiveName; Param = $cmdParm; runAs=$false }
    InstallExe $newTable
          
    if ($archiveSubTree) {
        $completeTempDestination = join-path $completeTempDestination $archiveSubTree
    }
    RobocopySourceDestination $completeTempDestination $completeDestination $copyAdditive

    rm -r $tempDir -Force -ErrorAction SilentlyContinue | Out-Null
}

function ExtractAllFromTarGz(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $sourceFile = $table["SourceFile"]
    $targzFileName = $table["TargzFileName"]
    $destination = $table["destination"]

    Write-Verbose "ExtractAllFromTarGz: targzFileName [$targzFileName] destination [$destination] in TargetFolder [$targetfolder]"
    if (-not $Execute) { 
        return 
    }

    $location = ResolveApplicationName -name git.exe -usePath $true
    if (-not $location) {
        throw "Unpacking the file [$targzFileName] requires extraction utility TAR.EXE.\n Make sure `"Git for Windows`" is installed on your machine."
    }

    $location = Split-Path $location -Parent
    $location = Split-Path $location -Parent

    $appDir = Join-Path $location "usr\bin"
    $completeApp = Join-Path $appDir "tar.exe"

    if (-not (Test-Path -path $completeApp -PathType Leaf)) {
        throw "Unpacking the file [$targzFileName] requires extraction utility [$completeApp].\n The utility wasn't found"
    }

    $completeDestination = Join-Path $destination $targzFileName

    Copy-Item $sourceFile $completeDestination -ErrorAction SilentlyContinue

    $dosCommand = @"
set path=$appDir;%PATH% & tar.exe -xz --force-local -f "$completeDestination" -C "$destination"
"@

    & cmd /c $dosCommand
    if ($LASTEXITCODE -gt 0) {
        throw "Running [$completeApp] Command failed with exit code $LASTEXITCODE"
    }
    
    Remove-Item "$completeDestination" -ErrorAction SilentlyContinue
}

function CreateBatch(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $filename = $table["Filename"]

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
"$AnacondaBasePath\Scripts\activate" "$AnacondaBasePath\envs\cntk-py34"
"@

    add-content -Path $filename -Encoding Ascii -Value $batchScript
}

function SetRegistryKey(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func = $table["Function"]
    $key  = $table["key"]
    $elevated = $table["Elevated"]

    if ($Execute) {
        $result = Test-Path -Path $key

        if (-not $result) {
            Write-Verbose "[$func]: [$key] will be created"
            if ($elevated) {
                $commandString = "& { new-item -Path '$key' }"
                RunPowershellCommand -command "$commandString" -elevated $true -maxErrorLevel 0
            }
            else {
                new-item -Path $key
            }
        }
    }
}

function SetRegistryKeyNameData(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func = $table["Function"]
    $key  = $table["key"]
    $regName  = $table["RegName"]
    $data = $table["data"]
    $dataType = $table["dataType"]
    $elevated = GetTableDefaultBool -table $table -entryName "Elevated" -defaultValue $true

    if ($Execute) {
        $tab = @{Function = "SetRegistryKey"; Key=$key; Elevated=$elevated}
        SetRegistryKey $tab
        
        Write-Verbose "[$func]: [$key]:[$regName] will be set to [$dataType]:[$data]"

        $commandString = "& { set-itemproperty -path '$key' -name '$regName'  -Type $dataType -Value $data }"
        RunPowershellCommand -command "$commandString" -elevated $elevated -maxErrorLevel 0

        
        #if ($elevated) {
        #    $commandString = "& { set-itemproperty -path '$key' -name '$regName'  -Type $dataType -Value $data }"
        #    RunPowershellCommand -command "$commandString" -elevated $true -maxErrorLevel 0
        #}
        #else {
        #    set-itemproperty -path '$key' -name '$regName'  -Type $dataType -Value $data
        #}
    }
}

function CreateBuildProtobufBatch(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func = $table["Function"]
    $filename = $table["FileName"]
    $sourceDir  = $table["SourceDir"]
    $targetDir  = $table["TargetDir"]
    $repoDirectory = $table["repoDirectory"]
        
    if ($Execute) {
        Remove-Item -Path $filename -ErrorAction SilentlyContinue | Out-Null

        $batchScript = GetBatchBuildProtoBuf $sourceDir $targetDir $repoDirectory

        add-content -Path $filename -Encoding Ascii -Value $batchScript
    }
}

function CreateBuildZlibBatch(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func = $table["Function"]
    $filename = $table["FileName"]
    $zlibSourceDir  = $table["zlibSourceDir"]
    $libzipSourceDir  = $table["libzipSourceDir"]
    $targetDir  = $table["TargetDir"]
    $repoDirectory = $table["repoDirectory"]
        
    if ($Execute) {
        Remove-Item -Path $filename -ErrorAction SilentlyContinue | Out-Null

        $batchScript = GetBatchBuildZlibBuf $zlibSourceDir $libzipSourceDir $targetDir $repoDirectory

        add-content -Path $filename -Encoding Ascii -Value $batchScript
    }
}

function DoProcess(
    [bool] $doExecute = $false,
    [string] $command,
    [string] $param,
    [string] $workingDir = "",
    [boolean] $requiresRunAs = $false,
    [int] $maxErrorLevel = 0,
    [bool] $throwOnError = $true)
{
    Write-Verbose "start-process [$command] with [$param]"

    if (-not $DoExecute) {
         Write-Host  "DEMOMODE - setting Exit Code: 0"
         return
    }

    if (-not $workingDir) {
        $workingDir = $(get-location).Path
    }
    if ($requiresRunAs) {
        $process = start-process -FilePath $command -ArgumentList $param -Wait -PassThru -Verb runas -WorkingDirectory $workingDir
    }
    else {
        $process = start-process -FilePath $command -ArgumentList $param -Wait -PassThru -WorkingDirectory $workingDir
    }

    $eCode = ($process.ExitCode)

    if (-not $throwOnError) {
        if ($ecode -gt $maxErrorLevel) {
            Write-Verbose "Running [start-process $commandString $param] failed with exit code [$ecode]"
        }
        return
    }

    if ($ecode -gt $maxErrorLevel) {
        throw "Running [start-process $commandString $param] failed with exit code [$ecode]"
    }
}


function SetUserEnvironmentVariable(
    [Parameter(Mandatory=$true)][string] $name,
    [Parameter(Mandatory=$true)][string] $content)    
{
    Write-Verbose "SetUserEnvironmentVariable [$name] with [$content]"
    
    if ($Execute) {
        [environment]::SetEnvironmentVariable($name, $content, "USER")
    }    
}

function SetMachineEnvironmentVariable(
    [Parameter(Mandatory=$true)][string] $name,
    [Parameter(Mandatory=$true)][string] $content)
{
    Write-Verbose "SetMachineEnvironmentVariable [$name] with [$content]"
    
    if ($Execute) {
        $commandString = "& { [environment]::SetEnvironmentVariable(`"$name`", `"$content`", `"MACHINE`") }"
        RunPowershellCommand -command "$commandString" -elevated $true -maxErrorLevel 0
    }    
}

function RunPowershellCommand(
    [string] $commandString,
    [boolean] $elevated,
    [int] $maxErrorLevel)
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
        throw "Running DOS Command '$Command $Argument' failed with exit code $LASTEXITCODE"
    }
}

function GetBatchBuildProtoBuf(
    [string] $sourceDir,
    [string] $targetDir,
    [string] $repoDirectory)
{
    $batchFile = join-path $repoDirectory "tools\devInstall\Windows\buildProtoVS15.bat"
@"
call $batchFile $sourceDir $targetDir
"@
}

function GetBatchBuildZlibBuf(
    [string] $zlibSourceDir,
    [string] $libzipSourceDir,
    [string] $targetDir,
    [string] $repoDirectory)
{
    $batchFile = join-path $repoDirectory "tools\devInstall\Windows\buildZlibVS15.bat"

@"
call $batchFile $libzipSourceDir $zlibSourceDir $targetDir
"@
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