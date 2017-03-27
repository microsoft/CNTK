#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function PreReqOperations(
    [array] $actionList = @())
{
    $continueInstallation = $true
    foreach ($item in $actionList) {
        foreach ($prereqItem in $item.PreReq) {
            $continueInstallation = $false
            PreRequisiteItem $prereqItem
        }
    }
    if (-not $continueInstallation) {
        throw "Not all pre-requisites installed, installation terminated."
    }
    Write-Host "Checking pre-requisites finished"
    Write-Host
}

function PreRequisiteItem(
    [hashtable] $item)
{
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    Invoke-Expression $expr 
}

function PrereqInfoVS15(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    Write-Warning "

Installation of Visual Studio 2015 Update 3 is a pre-requisite before installation can continue.
Please check 
  https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-Windows
for more details.
"
}

function PrereqInfoCuda8(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    Write-Warning "

Installation of NVidia CUDA 8.0 is a pre-requisite before installation can continue.
Please check 
  https://github.com/Microsoft/CNTK/wiki/Setup-CNTK-on-Windows
for more details.
"
}

function StopInstallation(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    throw "Not all pre-requisites installed, installation terminated."
}

# vim:set expandtab shiftwidth=4 tabstop=4: