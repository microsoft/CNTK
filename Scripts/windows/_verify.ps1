function VerifyOperations()
{
    Write-Host "Determining Operations to perform"

    foreach ($item in $operations) {
        if ($item.Ignore | ?{$global:installTarget -contains $_}) {
            continue
        }
        $needsInstall = $false

        foreach ($verificationItem in $item.Verification) {
            if ($item.Ignore | ?{$global:installTarget -contains $_}) {
                continue
            }
            $needsInstall = VerifyItem $verificationItem
            if (-not $needsInstall) {
                $global:operationList += $item
                break
            }
        }
    }

    Write-Host "List of Operations to be performed"
    $($global:operationList).Info
    #foreach ($item in $global:operationList) {
    #    $info = $item.Info
    #    Write-Host "    $info"
    #}
    Write-Host
}

function VerifyItem(
    [hashtable] $item
){
    $func = $item["Function"]
    $name = $item["Name"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]: [$name]"
    $result = Invoke-Expression $expr 

    return $result
}

function VerifyDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $result = (test-path -path $path -PathType Container)

    Write-Verbose "[$func]: [$path] returned [$result]"
    
    return $result
}

function VerifyWheelDirectory(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["WheelDirectory"]
    $forceUpdate = $table["ForceUpdate"]

    if ($forceUpdate) {
        $result = $false
    }
    else {
        $result = (test-path -path $path -PathType Container)
    }

    Write-Verbose "[$func]: [$path] returned [$result]"
    
    return $result
}

function VerifyFile(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $path = $table["Path"]

    $result = (test-path -path $path -PathType Leaf)

    Write-Verbose "[$func]: [$path] returned [$result]"
    
    return $result
}

function VerifyRegistryKey(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func = $table["Function"]
    $key = $table["Key"]

    $result = (test-path -path $key)

    Write-Verbose "[$func]: [$key] returned [$result]"
    
    return $result
}

function VerifyRegistryKeyName(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func     = $table["Function"]
    $key      = $table["Key"]
    $regName  = $table["RegName"]

    $result = Test-ItemProperty -Path $key -Name $regName

    Write-Verbose "[$func]: [$key]:[$regname] returned [$result]"
    
    return $result
}

function VerifyRegistryKeyNameData(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table

    $func     = $table["Function"]
    $key      = $table["Key"]
    $regName  = $table["RegName"]
    $regData  = $table["RegData"]

    $result = (test-path -path $key)

    if ($result) {
        $theKeyObj = get-item $key
        $result = ($theKeyObj.GetValue("$regName") -eq $regData)
    }

    Write-Verbose "[$func]: [$key]:[$regname] == [$regData] returned [$result]"
    
    return $result
}

function Test-ItemProperty (
    [string] $Path, 
    [string] $Name
)
{
    if (Test-Path $Path) {
        try {
            $ItemProperty = Get-ItemProperty -Path $Path -Name $Name -ErrorAction SilentlyContinue
            if ( $ItemProperty -ne $null ) {
                return $true 
            }
        }
        catch {
            return $false
        }
    }
    return $false
}