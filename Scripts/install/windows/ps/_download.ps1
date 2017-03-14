#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

function DownloadOperations()
{
    Write-Host "Performing download operations"

    foreach ($item in $Script:operationList) {
        foreach ($downloadItem in $item.Download) {
            DownloadItem $downloadItem
        }
    }

    Write-Host "Download operations finished"
    Write-Host
}


function DownloadItem(
    [hashtable] $item)
{
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    Invoke-Expression $expr 
}


function Download(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $source = $table["Source"]
    $destination = $table["Destination"]

    $downloadOk = Copy-FileWebRequest -SourceFile $source -OutFile $destination -maxtry 2

    if (-not $downloadOk) {
        throw "Download $SourceFile Failed!"
    }
}
