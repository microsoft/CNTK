function ActionOperations()
{
    Write-Host "Performing install operations"

    foreach ($item in $global:operationList) {
        
        foreach ($actionItem in $item.Action) {
            ActionItem $actionItem
        }
    }

    Write-Host "Install operations finished"
    Write-Host
}

function ActionItem(
    [hashtable] $item
){
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    $result = Invoke-Expression $expr 
    if (-not $result) {
        return 
    }
    return 
}


function InstallExe(
    [Parameter(Mandatory = $true)][hashtable] $table
)
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

    if ($runAs -eq $null) {
        $runAs = $true
    }
    if ($platform -ne $null) {
        $runningOn = ((Get-WmiObject -class Win32_OperatingSystem).Caption).ToUpper()
        $platform  = ($platform.ToString()).ToUpper()

        if (-not $runningOn.StartsWith($platform))
        {
            return
        }
    }

    if ($message -ne $null) {
        Write-Host $message
    }
    
    if ($dir -eq $null) {
        $ecode = DoProcess -command $cmd -param "$param" -requiresRunAs $runAs
    }
    else {
        $ecode = DoProcess -command $cmd -param "$param" -requiresRunAs $runAs -workingDir "$dir" 
    }
    
    if ( ($processWait -ne $null) -and ($Execute) -and ($false) ) {
        do {
    	    start-sleep 20
	        $pwait = Get-Process $processWait -ErrorAction SilentlyContinue
        } while (-not ($pwait -eq $null))
    }
    
      
    if ($ecode -eq 0) { return $true }
          
    return $false
}

function InstallWheel(
    [Parameter(Mandatory = $true)][hashtable] $table
)
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
    $newPaths = Invoke-DosCommand $condaExe (Write-Output ..activate cmd.exe $EnvName)

    $oldPath = $env:PATH
    $env:PATH = $newPaths + ';' + $env:PATH
    if (test-path $whlDirectory -PathType Container) {
        Invoke-DosCommand pip (Write-Output uninstall cntk --yes)
    }

    Invoke-DosCommand pip (Write-Output install $whl)
    $env:PATH = $oldPath 
    return
}

function MakeDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    
    $func = $table["Function"]
    $path = $table["Path"]

    if (-not (test-path -path $path)) {
        if ($Execute) {
            New-Item $path -type directory
        }
    }
    
    return $true
}

function AddToPath(
    [Parameter(Mandatory = $true)][hashtable] $table
)
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
        return $true
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
    return $true
}

function ExtractAllFromZip(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func    = $table["Function"]
    $zipFileName = $table["zipFileName"]
    $destinationFolder = $table["destinationFolder"]

    if (-not (test-path -path $destinationFolder)) {
        return $false
    }
    if (-not (test-path $zipFileName -PathType Leaf)) {
        return $false
    }

    if ($Execute) {
        $obj = new-object -com shell.application
        $zipFile = $obj.NameSpace($zipFileName)
        $destination = $obj.NameSpace($destinationFolder)

        $destination.CopyHere($zipFile.Items())
    }
    return $true
}

function DoProcess(
    [string]  $command,
    [string]  $param,
    [string]  $workingDir = "",
    [boolean] $requiresRunAs = $false)
{
    $info = "start-process [$command] with [$param]"

    Write-Verbose "$info"

    if (-not $Execute) {
         Write-Host  "** Running in DEMOMODE - setting Exit Code **: 0"
         return 0
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

    if ($eCode -ne 0) {
        Write-Host  "$message ** Exit Code **:($eCode)"
    } else {
        Write-Verbose "$message ** Exit Code **:($eCode)"
    }
    return $eCode
}



function SetEnvVar(
    [Parameter(Mandatory=$true)][string] $name,
    [Parameter(Mandatory=$true)][string] $content,
    [string] $location = "Machine")
{
    Write-Verbose "SetEnvVar [$name] with [$content]"
    
    if ($Execute) {
        # [environment]::SetEnvironmentVariable($name, $content, $location)

        $commandString = "& { [environment]::SetEnvironmentVariable('"+$name+"', '"+$content+"', '"+$location+"') }"

        RunPowershellCommand -command "$commandString" -elevated $true
    }    
}

function RunPowershellCommand(
    [string] $commandString,
    [boolean] $elevated
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
    return ($ecode -eq 0)
}

function Invoke-DosCommand {
  [CmdletBinding()]
  Param(
    [ValidateScript({ Get-Command $_ })]
    [string] $Command,
    [string[]] $Argument,
    [string] [ValidateScript({ Test-Path -PathType Container $_ })] $WorkingDirectory,
    [switch] $IgnoreNonZeroExitCode,
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
    if (($LASTEXITCODE -ne 0) -and -not $IgnoreNonZeroExitCode) {
        throw "Running '$Command $Argument' failed with exit code $LASTEXITCODE"
    }
}