#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#
function OpAnaconda3411(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $AnacondaBasePath)
{
    $targetFolder = Split-Path $AnacondaBasePath -Parent
    $prodSubDir = Split-Path $AnacondaBasePath -Leaf
    $prodName = "Anaconda3-4.1.1"
    $prodFile = "Anaconda3-4.1.1-Windows-x86_64.exe"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://repo.continuum.io/archive/Anaconda3-4.1.1-Windows-x86_64.exe"
    $downloadSize = 370055720

    @( @{ShortName = "ANA3-411"; Name = $prodName;  VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath; } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize } );
         # command line parameters for Anaconda installer: /D=$targetPath must be the last parameter and can not be surrounded by quotes
         Action = @( @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$targetPath"; runAs=$false; Message = ".... This will take some time. Please be patient ...." } );
     } )
}

function OpAnacondaEnv(
    [parameter(Mandatory=$true)][string] $AnacondaBasePath,
    [parameter(Mandatory=$true)][string] $repoDir,
    [parameter(Mandatory=$true)][string] $reponame,
    [parameter(Mandatory=$true)][string] $pyVersion)
{
    $prodName = "Python $pyVersion Environment"
    $targetFolder = Split-Path $AnacondaBasePath -Parent
    $prodSubDir = Split-Path $AnacondaBasePath -Leaf
    $targetPath = join-path $targetFolder $prodSubDir
    $envName = "cntkdev-py$pyVersion"
    $envDir = "envs\$envName"
    $envVar = "CNTK_PY$($pyVersion)_PATH";
    $envValue = join-path $targetPath $envDir

    $ymlDirectory = join-path $repoDir $repoName
    $ymlDirectory = join-path $ymlDirectory "scripts\install\windows"

    @{ ShortName = "PYENV"; Name = $prodName;  VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Creating $prodName";
      Verification  = @( @{Function = "VerifyRunAlways" } );
      Action = @( @{Function = "InstallYml"; BasePath = $targetPath; Env = $envName; ymlFile= "$ymlDirectory\conda-windows-cntk-py$($pyVersion)-environment.yml" },
                  @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } )
     }
}

function OpBoost160VS15(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "Boost 1.60.0"
    $prodFile = "boost_1_60_0-msvc-14.0-64.exe"
    $prodSubDir = "boost_1_60_0-msvc-14.0"
    $targetPath = join-path $targetFolder $prodSubDir
    $downloadSource = "https://sourceforge.net/projects/boost/files/boost-binaries/1.60.0/boost_1_60_0-msvc-14.0-64.exe/download"
    $envVar = "BOOST_INCLUDE_PATH"
    $envVarLib = "BOOST_LIB_PATH"
    $envContentLib = "$targetPath\lib64-msvc-14.0"
    $downloadSize = 0

    @( @{Name = $prodName; ShortName = "BOOST160VS15"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $targetPath },
                        @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVarLib; Content  = $envContentLib } );
        Download = @( @{Function = "Download"; Method = "WebClient"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize } );
        Action = @( @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content = $targetPath },
                    @{Function = "SetEnvironmentVariable"; EnvVar = $envVarLib; Content  = $envContentLib },
                    @{Function = "InstallExe"; Command = "$cache\$prodFile"; Param = "/dir=`"$targetPath`" /SP- /SILENT /NORESTART"; runAs=$false } );
     } )
}

function OpCMake362(
    [parameter(Mandatory=$true)][string] $cache)
{
    $prodName = "CMake 3.6.2"
    $targetPath = join-path $env:ProgramFiles "cmake\bin"
    $cmakeName = "cmake-3.6.2-win64-x64.msi"
    $downloadSource = "https://cmake.org/files/v3.6/cmake-3.6.2-win64-x64.msi"
    $downloadSize = 15771063
    
    @( @{ShortName = "CMake362"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
        Verification = @( @{Function = "VerifyWinProductExists"; Match = "^CMake$"; Version = "3.6.2" }  );
        Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$cmakeName"; ExpectedSize = $downloadSize } );
        Action = @( @{Function = "InstallMsi"; MsiName =  "$cmakeName" ; MsiDir   = "$cache" }  ,
                    @{Function = "AddToPath"; Dir = "$targetPath" } );
        } )
}

function OpCNTKMKL3
    ([parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "CNTK Custom MKL Version 3"
    $prodFile = "CNTKCustomMKL-Windows-3.zip"
    $prodSubDir = "CNTKCustomMKL"
    $targetPath = join-path $targetFolder $prodSubDir
    $targetPathCurrenVersion = join-path $targetPath "3"
    $envVar = "CNTK_MKL_PATH";
    $envValue = $targetPath
    $downloadSource = "https://www.cntk.ai/mkl/$prodFile";
    $downloadSize = 10331909

    @(  @{ShortName = "CNTKMKL3"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPathCurrenVersion"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyDirectory"; Path = $targetPathCurrenVersion },
                            @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
          Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize } );
          Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; destinationFolder = $prodSubDir },
                      @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpMSMPI70([parameter(
    Mandatory=$true)][string] $cache)
{
    $remoteFilename = "MSMpiSetup.exe"
    $localFilename = "MSMpiSetup70.exe"
    $downloadSource = "https://download.microsoft.com/download/D/7/B/D7BBA00F-71B7-436B-80BC-4D22F2EE9862/$remoteFilename";
    $downloadSize = 5277808

    @( @{Name = "MSMPI Installation"; ShortName = "CNTK"; VerifyInfo = "Checking for installed MSMPI 70"; ActionInfo = "Installing MSMPI 70";
         Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft MPI \(\d+\."; Version = "7.0.12437.6"; MatchExact = $false } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$localFilename"; ExpectedSize = $downloadSize } );
         Action = @( @{Function = "InstallExe"; Command =  "$cache\$localFilename" ; Param = "/unattend" } )
        } )
}

function OpMSMPI70SDK(
    [parameter(Mandatory=$true)][string] $cache)
{
    $remoteFilename = "msmpisdk.msi"
    $localFilename = "msmpisdk70.msi"
    $downloadSource = "https://download.microsoft.com/download/D/7/B/D7BBA00F-71B7-436B-80BC-4D22F2EE9862/$remoteFilename";
    $downloadSize = 2285568  

    @( @{Name = "MSMPI SDK70 Installation"; ShortName = "CNTK"; VerifyInfo = "Checking for installed MSMPI 70 SDK"; ActionInfo = "Installing MSMPI 70 SDK";
         Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft MPI SDK \(\d+\."; Version = "7.0.12437.6"; MatchExact = $false } );
         #Verification = @( @{Function = "VerifyWinProductExists"; Match = "^Microsoft MPI SDK \(\d+\."; Compare = "^Microsoft MPI SDK \(7\.0\.12437\.6\)"; MatchExact = $false } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$localFilename"; ExpectedSize = $downloadSize } );
         Action = @( @{Function = "InstallMsi"; MsiName = "$localFilename" ; MsiDir = "$cache" } )
        } )
}

function OpNvidiaCub141(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "NVidia CUB 1.4.1"
    $prodFile = "cub-1.4.1.zip"
    $prodSubDir = "cub-1.4.1"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "CUB_PATH";
    $envValue = $targetPath
    $downloadSource = "https://codeload.github.com/NVlabs/cub/zip/1.4.1";
    $downloadSize = 51376046

    @( @{ShortName = "CUB141"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = "$targetFolder"; destinationFolder = $prodSubDir; zipSubTree= $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpNVidiaCudnn5180(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "NVidia CUDNN 5.1 for CUDA 8.0"
    $cudnnWin7 = "cudnn-8.0-windows7-x64-v5.1.zip"
    $cudnnWin10 = "cudnn-8.0-windows10-x64-v5.1.zip"

    $prodSubDir =  "cudnn-8.0-v5.1"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "CUDNN_PATH"
    $envValue = join-path $targetPath "cuda"
    $downloadSource = "http://developer.download.nvidia.com/compute/redist/cudnn/v5.1"
    $downloadSizeWin7 = 0
    $downloadSizeWin10 = 0

    @( @{ShortName = "CUDNN5180"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyDirectory"; Path = $envValue },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{Function = "DownloadForPlatform"; Method = "WebRequest"; Platform = "^Microsoft Windows 7"; Source = "$downloadSource/$cudnnWin7"; Destination = "$cache\$cudnnWin7"; ExpectedSize = $downloadSizeWin7 },
                       @{Function = "DownloadForPlatform"; Method = "WebRequest"; Platform = "^Microsoft Windows (8|10|Server 2008 R2|Server 2012 R2)"; Source = "$downloadSource/$cudnnWin10"; Destination = "$cache\$cudnnWin10"; ExpectedSize = $downloadSizeWin10 } );
         Action = @( @{Function = "ExtractAllFromZipForPlatform"; Platform = "^Microsoft Windows 7"; zipFileName = "$cache\$cudnnWin10"; destination = $targetFolder; destinationFolder = $prodSubDir },
                     @{Function = "ExtractAllFromZipForPlatform"; Platform = "^Microsoft Windows (8|10|Server 2008 R2|Server 2012 R2)"; zipFileName = "$cache\$cudnnWin10"; destination = $targetFolder; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
         })
}

function OpOpenCV31(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "OpenCV-3.1"
    $prodFile = "opencv-3.1.0.exe"
    $prodSubDir = "Opencv3.1.0"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "OPENCV_PATH_V31";
    $envValue = "$targetPath\build"
    $downloadSource = "https://netcologne.dl.sourceforge.net/project/opencvlibrary/opencv-win/3.1.0/opencv-3.1.0.exe"
    $downloadSize = 0
    $archiveSubTree = "opencv"

    @(  @{ShortName = "OPENCV310"; Name = $prodName; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
          Verification = @( @{Function = "VerifyDirectory"; Path = "$targetPath" },
                            @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
          Download = @( @{ Function = "Download"; Method = "WebClient"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize } );
          Action = @( @{Function = "Extract7zipSelfExtractingArchive"; archiveName = "$cache\$prodFile"; destination = "$targetFolder"; destinationFolder = $prodSubDir; archiveSubTree= $archiveSubTree },
                      @{Function = "SetEnvironmentVariable"; EnvVar= $envVar; Content = $envValue } );
         } )
}

function OpProtoBuf310VS15(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder,
    [parameter(Mandatory=$true)][string] $repoDirectory)
{
    # unzip protobuf source in $protoSourceDir = $targetfolder\src\$prodsubdir
    # create batch file to build protobuf files in $scriptDirectory = $targetFolder\script
    # the script file can be used to create the compiled protobuf libraries in $targetPath = $targetFolder\$prodSubDir

    $prodName = "ProtoBuf 3.1.0 Source"
    $prodSrcSubdir = "protobuf-3.1.0"
    $prodFile = "protobuf310.zip"
    $prodSubDir =  "protobuf-3.1.0-vs15"
    $batchFile = "buildProtoVS15.cmd"

    $protoSourceDir = join-path $targetFolder "src"
    $targetPath = Join-Path $protoSourceDir $prodSrcSubdir
    $scriptDirectory = join-path $targetFolder "script"
    $buildDir = join-path $targetFolder $prodSubDir
    $downloadSource = "https://github.com/google/protobuf/archive/v3.1.0.zip"
    $downloadSize = 5648581    

    @( @{ShortName = "PROTO310VS15"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize} );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $protoSourceDir; zipSubTree = $prodSrcSubdir; destinationFolder = $prodSrcSubdir },
                     @{Function = "MakeDirectory"; Path = $scriptDirectory },
                     @{Function = "CreateBuildProtobufBatch"; FileName = "$scriptDirectory\$batchFile"; SourceDir = $targetPath; TargetDir = $buildDir; RepoDirectory = $repoDirectory } );
        } )
}

function OpProtoBuf310VS15Prebuild(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "ProtoBuf 3.1.0 VS15 CNTK Prebuild"
    $prodFile = "protobuf-3.1.0-vs15.zip"
    $prodSubDir =  "protobuf-3.1.0-vs15"

    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "PROTOBUF_PATH"
    $envValue = $targetPath
    $downloadSource = "https://cntk.ai/binarydrop/prerequisites/protobuf/protobuf-3.1.0-vs15.zip"
    $downloadSize = 0    

    @( @{ShortName = "PROTO310VS15PRE"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize} );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree = $prodSubDir; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue }  );
        } )
}

function OpScanProgram()
{
    @{ ShortName = "SCANPROG"; VerifyInfo = "Scan System for installed programs";
      Verification = @( @{Function = "VerifyScanPrograms" } )
     }
}

function OpSwig3010(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "SWIG 3.0.10"
    $prodFile = "swigwin-3.0.10.zip"
    $prodSubDir =  "swigwin-3.0.10"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "SWIG_PATH"
    $envValue = $targetPath
    $downloadSource = "http://prdownloads.sourceforge.net/project/swig/swigwin/swigwin-3.0.10/swigwin-3.0.10.zip"
    $downloadSize = 0

    @( @{ShortName = "SWIG3010"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName";  
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Method = "WebClient"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree =$prodSubDir; destinationFolder =$prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue } );
        } )
}

function OpCheckVS15Update3
{
    @( @{Name = "Verify Installation of VS2015, Update 3"; ShortName = "PREVS15U3"; VerifyInfo = "Checking for Visual Studio 2015, Update 3"; 
                        Verification = @( @{Function = "VerifyWinProductVersion"; Match = "^Microsoft Build Tools 14.0 \(amd64\)$"; Version = "14.0.25420"; MatchExact = $true} ); 
                        PreReq = @( @{Function = "PrereqInfoVS15" } );
                        Action = @( @{Function = "StopInstallation" } )
                        } )
}

function OpCheckCuda8
{
    $programPath = join-path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v8.0"
    @( @{Name = "Verify Installation of NVidia Cuda 8"; ShortName = "PRECUDA8"; VerifyInfo = "Checking for NVidia Cuda 8"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $programPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = "CUDA_PATH_V8_0"; Content = $programPath } ); 
         PreReq = @( @{Function = "PrereqInfoCuda8" } );
         Action = @( @{Function = "StopInstallation" } )
        } )
}

function OpZlibVS15(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder,
    [parameter(Mandatory=$true)][string] $repoDirectory)
{
    # unzip protobuf source in $protoSourceDir = $targetfolder\src\$prodsubdir
    # create batch file to build protobuf files in $scriptDirectory = $targetFolder\script
    # the script file can be used to create the compiled protobuf libraries in $targetPath = $targetFolder\$prodSubDir

    $prodName = "zlib / libzip from source"
    $zlibProdName = "zlib-1.2.8"
    $zlibFilename = "zlib128.zip" 
    # $zlibDownloadSource = "https://netix.dl.sourceforge.net/project/libpng/zlib/1.2.8/zlib128.zip"
    $zlibDownloadSource = "https://cntk.ai/binarydrop/prerequisites/zip/zlib128.zip"
    $downloadSizeZlib = 0
    
    $libzipProdName = "libzip-1.1.3"
    $libzipFilename = "libzip-1.1.3.tar.gz" 
    $libzipDownloadSource = "https://nih.at/libzip/libzip-1.1.3.tar.gz"
    $downloadeSizeLibzip = 0
    
    $prodSubDir =  "zlib-vs15"
    $batchFile = "buildZlibVS15.cmd"

    $sourceCodeDir = join-path $targetFolder "src"
    $scriptDirectory = join-path $targetFolder "script"
    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "ZLIB_PATH"
    $envValue = $targetPath
    
    @( @{ShortName = "ZLIBVS15"; VerifyInfo = "Checking for $prodName in $sourceCodeDir"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = "$sourceCodeDir\$zlibProdName" },
                           @{Function = "VerifyDirectory"; Path = "$sourceCodeDir\$libzipProdName" },
                           @{Function = "VerifyFile"; Path = "$scriptDirectory\$batchFile" } );
         Download = @( @{ Function = "Download"; Source = $zlibDownloadSource; Destination = "$cache\$zlibFilename"; ExpectedSize = $downloadSizeZlib }, 
                       @{ Function = "Download"; Source = $libzipDownloadSource; Destination = "$cache\$libzipFilename"; ExpectedSize = $downloadeSizeLibzip } );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$zlibFilename"; destination = $sourceCodeDir; zipSubTree =$zlibProdName; destinationFolder =$zlibProdName },
                     @{Function = "ExtractAllFromTarGz"; SourceFile =  "$cache\$libzipFilename"; TargzFileName = "$libzipFilename"; destination = $sourceCodeDir },
                     @{Function = "MakeDirectory"; Path = $scriptDirectory },
                     @{Function = "CreateBuildZlibBatch"; FileName = "$scriptDirectory\$batchFile"; zlibSourceDir = (join-path $sourceCodeDir $zlibProdName); libzipSourceDir = (join-path $sourceCodeDir $libzipProdName); TargetDir = $targetPath; RepoDirectory = $repoDirectory } );
        } )
}

function OpZlibVS15Prebuild(
    [parameter(Mandatory=$true)][string] $cache,
    [parameter(Mandatory=$true)][string] $targetFolder)
{
    $prodName = "ZLib VS15 CNTK Prebuild"
    $prodFile = "zlib-vs15.zip"
    $prodSubDir =  "zlib-vs15"


    $targetPath = join-path $targetFolder $prodSubDir
    $envVar = "ZLIB_PATH"
    $envValue = $targetPath
    $downloadSource = "https://cntk.ai/binarydrop/prerequisites/zip/zlib-vs15.zip"
    $downloadSize = 0    

    @( @{ShortName = "ZLIBVS15PRE"; VerifyInfo = "Checking for $prodName in $targetPath"; ActionInfo = "Installing $prodName"; 
         Verification = @( @{Function = "VerifyDirectory"; Path = $targetPath },
                           @{Function = "VerifyEnvironmentAndData"; EnvVar = $envVar; Content = $envValue } );
         Download = @( @{ Function = "Download"; Method = "WebRequest"; Source = $downloadSource; Destination = "$cache\$prodFile"; ExpectedSize = $downloadSize} );
         Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$cache\$prodFile"; destination = $targetFolder; zipSubTree = $prodSubDir; destinationFolder = $prodSubDir },
                     @{Function = "SetEnvironmentVariable"; EnvVar = $envVar; Content  = $envValue }  );
        } )
}


# vim:set expandtab shiftwidth=2 tabstop=2: