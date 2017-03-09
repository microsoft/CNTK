#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function DownloadOperations(
    [parameter(Mandatory=$true)][array] $downloadList)
{
    Write-Host "Performing download operations"

    foreach ($item in $downloadList) {
        foreach ($downloadItem in $item.Download) {
            DownloadItem $downloadItem
        }
    }

    Write-Host "Download operations finished"
    Write-Host
}


function DownloadItem(
    [hashtable] $item
)
{
    $func = $item["Function"]

    $expr = $func +' $item' 
        
    Write-Verbose "Calling Operation: [$func]"
    $result = Invoke-Expression $expr 
}

function DownloadForPlatform(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $platform = $table["Platform"]

    if (PlatformMatching $platform) {
        Download $table
    }
}

function Download(
    [Parameter(Mandatory = $true)][hashtable] $table)
{
    FunctionIntro $table

    $func = $table["Function"]
    $source = $table["Source"]
    $method = GetTableDefaultString -table $table -entryName "Method" -defaultValue "WebRequest"
    $userAgent = GetTableDefaultString -table $table -entryName "UserAgent" -defaultValue = "InternetExplorer"
    $destination = $table["Destination"]
    $expectedHash = GetTableDefaultInt -table $table -entryName "expectedHash" -defaultValue ""

    if (test-path $destination -PathType Leaf) {
        Write-Host File [$destination] already exists
        return
    }

    if ($method -eq "WebRequest") {
        DownloadFileWebRequest -SourceFile $source -OutFile $destination -UserAgent $userAgent -expectedHash $expectedHash
    }
    else {
        DownloadFileWebClient -SourceFile $source -OutFile $destination -expectedHash $expectedHash
    }
}

function LocalCopyFile(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
    FunctionIntro $table
    $func = $table["Function"]
    $source = $table["Source"]
    $destination = $table["Destination"]

    if (-not $Execute) {
         Write-Host  "$message ** Running in DEMOMODE - no download performed"
         return $true
    }
    if (test-path $destination -PathType Leaf) {
        Write-Host File [$destination] already exists
        return
    }
    if (-not (test-path $source -PathType Leaf)) {
        throw "Sourcefile [$source] is missing"
    }
    
    Write-Host Copying [$source] to local disk ...
    new-item $destination -type File -Force -ErrorAction SilentlyContinue
    copy-Item $source $destination -Force -ErrorAction SilentlyContinue
}

function NotImplemented(
    [Parameter(Mandatory = $true)][hashtable] $table
)
{
     throw "Call to function 'NotImplemented' "
}

function DownloadAndExtract(
    [string] $tPath,
    [string] $sAddress,
    [string] $fileName,
    [string] $targetPathRoot
){
    $outFileName  = Join-Path $tPath $fileName

    DownloadFileWebClient -SourceFile $sAddress `
                    -OutFile $outFileName `
                    -tempFileName $fileName

    Write-Host Extracting into targetpath
    Write-Host

    ExtractAllFromZip $outFileName $targetPathRoot
}


function DownloadFileWebRequest (
    [string] $SourceFile,
    [string] $OutFile,
    [string] $userAgent,
    [string] $expectedHash)
{
    Write-Host "Downloading [$SourceFile], please be patient...."
    if (-not $Execute) {
         Write-Host  "$message ** Running in DEMOMODE - no download performed"
         return $true
    }

    if (test-path -path $outFile) {
        Write-Host "File [$outFile] already exists"
        return $true
    }   

    $TempFile = [System.IO.Path]::GetTempFileName()
    try {
        $response = Invoke-WebRequest -Uri $SourceFile -OutFile $TempFile -UserAgent $userAgent -TimeoutSec 120 
    } 
    catch {
      $errorCode = $_.Exception.Response.StatusCode.Value__

      Remove-Item -path $TempFile -ErrorAction SilentlyContinue
      throw "Download $SourceFile Failed! WebRequest reported errorCode: [$errorCode]"
    }

    CheckHash $TempFile $expectedHash

    new-item $outFile -type File -Force -ErrorAction SilentlyContinue
    move-Item $TempFile $OutFile -Force -ErrorAction SilentlyContinue

    if (-not (Test-Path -Path $OutFile)) {
        throw "Download $SourceFile Failed!"
    }
    # we have a file with our expected filename, we are in good shape and ready to report success
    # in case the above rename failed, but the target file exist, we clean up a possible dangling TempFile
    # no need to check if this file is really there. we don't care if it succeeds or not
    Remove-Item -path $TempFile -ErrorAction SilentlyContinue
}

function DownloadFileWebClient(
    [string] $SourceFile,
    [string] $OutFile,
    [string] $expectedHash,
    [int] $timeout = 600,
    [int] $maxtry = 5)
{
    $sb ={
            param([string]$uri,[string]$outfile)
            (New-Object System.Net.WebClient).DownloadFile($uri,$outfile) 
         }
    #---------------

    $startTime = Get-Date
    Write-Host "Downloading [$SourceFile], please be patient, no progress message is shown ..."
    if (-not $Execute) {
         Write-Host  "$message ** Running in DEMOMODE - no download performed"
         return
    }

    $TempFile = [System.IO.Path]::GetTempFileName()

    for ($count=1; $count -le $maxtry; $count +=1) {
        if ($count -gt 1) {
            Write-Host "Iteration [$count] of [$maxtry]"
        }
        
        if ($count -gt 1) {
            start-sleep -Seconds 5
        }
        if (Test-Path -Path $TempFile) {
            # the file we temporary use as a download cache could exist
            # if it does, we remove it and terminate if this fails
            Remove-Item -path $TempFile -ErrorAction Stop
        }    
        
        if (test-path -path $outFile) {
            Write-Host "File [$outFile] already exists"
            return
        }   

        $job = start-job -scriptblock $sb -ArgumentList $sourceFile, $TempFile
        Wait-Job $job -Timeout $timeout

        $jState = $job.State.ToUpper()
        $jStart = $job.PSBeginTime
        $jEnd = $job.PSEndTime
        $jError =  $job.ChildJobs[0].Error
        $current = Get-Date

        switch ($jState) {
            "COMPLETED" { 
                if ($jError.Count -eq 0) {
                    Write-Verbose "End binary download!"

                    Remove-Job $job -force -ErrorAction SilentlyContinue

                    CheckHash $TempFile $expectedHash

                    # we now have the temporary file, we need to rename it
                    new-item $outFile -type File -Force -ErrorAction SilentlyContinue
                    move-Item $TempFile $OutFile -Force -ErrorAction SilentlyContinue

                    if (Test-Path -Path $OutFile) {
                        # we have a file with our expected filename, we are in good shape and ready to report success
                        # in case the above rename failed, but the target file exist, we clean up a possible dangling TempFile
                        # no need to check if this file is really there. we don't care if it succeeds or not
                        Remove-Item -path $TempFile -ErrorAction SilentlyContinue

                        return
                    }

                    # we got here because we finished the job, but some operation failed (i.e. the rename above. we can just try again)
                    Write-Verbose "Job completed but rename operation failed, retrying..."
                    continue
                }

                Write-Host "Job Completed with Error: [$jStart] to [$current]"
                Write-Host $jError
            }
            "RUNNING"   {
                Write-Host "Job Timeout: [$jStart] to [$current]"
            }
            "FAILED"    {
                $current = Get-Date
                Write-Host "Job Failed: [$jStart] to [$current]"
                Write-Host "Error: $jError"
            }
            default     {
                Write-Host "Job State: [$Error] - [$jStart] to [$current]"
            }
        }
        Remove-Job $job -force -ErrorAction SilentlyContinue
    }

    throw "Download $SourceFile Failed!"
}

function PlatformMatching(
    [string] $regExprPlatform)
{
    $runningOn = ((Get-WmiObject -class Win32_OperatingSystem).Caption).ToUpper()
    $isMatching = ($runningOn -match $regExprPlatform) 

    Write-Verbose "Function [PlatformMatching]: $runningOn -match on platform [$regExprPlatform] = [$isMatching]"
    return $isMatching
}

function CheckHash(
    [string] $tempFile,
    [string] $expectedHash)
{
    if ($expectedHash) {
        $fileHash = (Get-FileHash -Path $tempFile -Algorithm SHA256 -ErrorAction SilentlyContinue).Hash
        if ($fileHash -ne $expectedHash) {
              Write-Warning "Hash for downloaded file is not matching expected hash value."
              Write-Warning "This could be a download error or an unexpected version change at the download source."
              Write-Warning "The installation will continue ...."
        }
    }
}

# vim:set expandtab shiftwidth=4 tabstop=4: