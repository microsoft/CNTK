#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

$operations = @(
    @{Name = "Scan System for installed programs"; ShortName = "SCANPROG"; Info = "Scan System for installed programs";
      Verification = @( @{Function = "VerifyScanPrograms" } )
     },
    @{Name = "Verifying Installation contents"; ShortName = "INSTCONTENT"; Info = "Verifying Installation contents";
      Verification = @( @{Function = "VerifyInstallationContent"; Path = "$cntkRootDir" } )
     },
    @{Name = "Installation VS2015 Runtime"; ShortName = "VS2015"; Info = "Install VS2015 Runtime";
      Verification = @( @{Function = "VerifyWin32ProductExists"; Match = "^Microsoft Visual C\+\+ 201(5|7) x64 Additional Runtime" },
                        @{Function = "VerifyWin32ProductExists"; Match = "^Microsoft Visual C\+\+ 201(5|7) x64 Minimum Runtime" } );
      Action = @( @{Function = "InstallExe"; Command  = "$cntkRootDir\prerequisites\VS2015\vc_redist.x64.exe"; Param = "/install /passive /norestart"; Message="Installing VS2015 Runtime...." } )
     },
    @{Name = "MSMPI Installation"; ShortName = "CNTK"; Info = "Install MSMPI";
      Verification = @( @{Function = "VerifyWin32ProductVersion"; Match = "^Microsoft MPI \(\d+\."; Version = "7.0.12437.6" } );
      Action = @( @{Function = "InstallExe"; Command = "$cntkRootDir\prerequisites\msmpisetup.EXE"; Param = "/unattend"; Message="Installing MSMPI ...." } )
     },
    @{Name = "Anaconda3-4.1.1"; ShortName = "ANA3-411"; Info = "Install Anaconda3-4.1.10";
      Verification = @( @{Function = "VerifyDirectory"; Path = "$AnacondaBasePath"; } );
      Download = @( @{Function = "Download"; Source = "https://repo.continuum.io/archive/Anaconda3-4.1.1-Windows-x86_64.exe"; Destination = "$localCache\Anaconda3-4.1.1-Windows-x86_64.exe" } );
      Action = @( @{Function = "InstallExe"; Command = "$localCache\Anaconda3-4.1.1-Windows-x86_64.exe"; Param = "/InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=$AnacondaBasePath"; runAs=$false; Message="Installing Anaconda3-4.1.1. This will take several minutes. Please be patient ...."} );
     },
    @{Name = "CNTK Python Environment"; ShortName = "CNTKPY"; Info = "Setup CNTK PythonEnvironment $PyVersion";
      Verification  = @( @{Function = "VerifyRunAlways"  } );
      Action = @( @{Function = "InstallYml"; BasePath = $AnacondaBasePath; Env = "cntk-py$PyVersion"; ymlFile= "$MyDir\conda-windows-cntk-py$PyVersion-environment.yml"; PyVersion = $PyVersion } )
     },
    @{Name = "CNTK WHL Install"; ShortName = "CNTKWHL"; Info = "Setup/Update CNTK Wheel $PyVersion";
      Verification  = @( @{Function = "VerifyRunAlways" } );
      Action = @( @{Function = "InstallWheel"; BasePath = "$AnacondaBasePath"; EnvName = "cntk-py$PyVersion"; WheelDirectory="$AnacondaBasePath\envs\cntk-py$PyVersion\Lib\site-packages\cntk"; PyVersion = $PyVersion; Message="Setup/Update of CNTK Wheel $PyVersion environment. Please be patient...." } )
     },
    @{Name = "Create CNTKPY batch file"; ShortName = "BATCH"; Info = "Create CNTKPY batch file";
      Verification  = @( @{Function = "VerifyFile"; Path = "$cntkRootDir\scripts\cntkpy$PyVersion.bat"; PyVersion = $PyVersion } );
      Action = @( @{Function = "CreateBatch"; Filename = "$cntkRootDir\scripts\cntkpy$PyVersion.bat"; PyVersion = $PyVersion } )
     }
)
