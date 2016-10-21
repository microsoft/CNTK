CNTK V2 Setup and Installation
==============================

You have the choice of installing CNTK from binary distributions or from
the GitHub sources for both *Windows* and *Linux* environment with
optional support for Nvidia GPU. For users intending to use this
library, we recommend the binary distributions. We also provide scripts
for the binary installation for further convenience. Using CNTK from
GitHub sources is recommended for developers intending to contribute.

[Note: The CNTK v2 APIs are an alpha release meant for early users to
try the bits and provide feedback on the usability and functional
aspects of the API.]

**Language Support**:

CNTK V2 provides Python and C++ APIs. These APIs enables
programmatically defining CNTK models and drive their
training/evaluation, using either built-in data readers or user supplied
data in native Python numpy/C++ arrays.

*Python*:

-  CNTK V2 with Python APIs is supported natively on Windows with Python
   **3.4.4** and on Linux with Python **3.5**. For both the platforms
   one can create a 3.4.4 python environment within any Anaconda (Python
   2.7 or 3.x version).

*Brainscript*:

-  CNTK V2 also supports the BrainScript framework.

CNTK installation overview
--------------------------

You can install CNTK with in three different ways:

-  Binary install with prepackaged scripts (**Recommended**): This
   provides the fastest way to get started with CNTK. **See the steps
   below.**
-  `Binary install with manual
   steps <https://github.com/Microsoft/CNTK/wiki/CNTK-2.0-Setup#step-by-step-cntk-v2-installation>`__.
   Only for those who need highly customized installations (not
   recommended).
-  `Build from
   sources <https://github.com/Microsoft/CNTK/wiki/CNTK-2.0-Setup#build-from-sources>`__
   (for developers): Those interested in contributing to CNTK.

Binary installation with scripts (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two steps in CNTK V2 binary installation:

**Step 1:** Prerequisites

-  We provide scripts for both Windows and Linux (Ubuntu) platforms to
   install the pre-requisites. Refer to the descriptions below.

**Step 2:** Python (Anaconda environment) installation

-  We use the popular Anaconda as our Python environment of choice. This
   installation will preserve pre-existing settings and creates an
   environment specific to CNTK. The scripts provided automates the
   following steps for you:
-  Install Anaconda.
-  Create a conda environment for CNTK V2 along with the packages needed
   to run iPython tutorial notebooks.
-  Activate the conda environment for CNTK V2.
-  Install CNTK V2 binary package.

Windows installation steps
------------------------------

Please follow the steps below to install the binaries. The script will
additionally clone the CNTK repository into ``C:\repos\cntk``.

**Step 1**: Prepare to run PowerShell scripts:

-  Downloaded the scripts from
   `here <https://cntk.ai/pippackages/cntk2a4_WindowsBinaryInstall.zip>`__.
   Unzip/extract in a local folder say ``C:\scripts``. One can create
   the directory by executing ``mkdir C:\scripts`` from a Windows command
   prompt.
-  Open PowerShell with administrator priviledges
-  Click on Windows Start
-  Search for Windows
   `PowerShell <https://cntk.ai/jup/v2doc/pswin-noadmin.png>`__ (see
   below), Right click on the icon and select *Run as administrator*

.. image:: https://cntk.ai/jup/v2doc/ps-with-admin.png
    :width: 250px
    :align: center
    :alt: powershell menu

-  Type and run: ``set-executionpolicy -executionpolicy unrestricted``.
   Upon being prompted, *select A*

.. image:: https://cntk.ai/jup/v2doc/pswin-with-admin.png
    :width: 600px
    :align: center
    :alt: powershell window with admin priviledges

-  Close the PowerShell window
-  Start a new Windows PowerShell application similar to previous step
   (not in Administrator mode)

.. image:: https://cntk.ai/jup/v2doc/pswin-noadmin.png
    :width: 400px
    :align: center
    :alt: powershell window without admin priviledges

**Step 2**: Run PowerShell scripts

-  Run: ``cd [Path to your unzipped PS scripts]``
-  Choose a wheel file appropriate for your machine:

  -  CPU:
     https://cntk.ai/pippackages/cpu/cntk-2.0a4-cp34-cp34m-win\_amd64.whl
  -  GPU:
     https://cntk.ai/pippackages/gpu/cntk-2.0a4-cp34-cp34m-win\_amd64.whl

-  Run: ``.\install.ps1 -Execute -cntkWhl [Location of the Whl file]``
-  Note: at a later time, if you need to re-install the same or a different wheel package, please run: ``.\install.ps1 -Execute -ForceWheelUpdate [Location of the Whl file]``
   
**Step 3**: Run Python setup

Open a windows command window or an anaconda command window - Run:
``[Anaconda install folder root]\Scripts\activate cntk-py34``

-  If you want to deploy a new CNTK wheel package at a later time simply
   re-run:
-  Run: ``pip install –upgrade [Location of the wheel file]``

Windows install details
~~~~~~~~~~~~~~~~~~~~~~~

The script automates the following installation steps:

-  Pre-requisites:
  -  Visual C++ Redistributable Package for Visual Studio 2013
  -  Visual C++ Redistributable Package for Visual Studio 2012
  -  Microsoft MPI of version 7 (7.0.12437.6).
  -  For NVIDIA GPU systems: ensures the latest NVIDIA driver are
     installed

-  Python setup:
  -  Install Anaconda (can take a while)
  -  Create an Anaconda CNTK environment cntk-py34 with conda create and
     install packages to run iPython notebook (can take some time)
  -  Install CNTK V2 Python Packages using pip

-  Example / Tutorial:
  -  Git installation
  -  Clone the CNTK repositories into ``C:\repos\cntk``

Linux installation steps
----------------------------

Please follow the steps below to install the binaries. The script will
additionally clone the CNTK repository into
``/home/[USERNAME]/repos/cntk``. Note: we have tested the script on
Ubuntu 16.0.4. We first summarize what the script installs for you and
sets the environment variables.

**Step 1**: Prepare to run scripts: - Downloaded the scripts from
`here <https://cntk.ai/pippackages/cntk2a4_LinuxBinaryInstall.zip>`__.
Create a directory under ``/home/[USERNAME]`` and unzip the scripts in
that folder.

**Step 2**: Run the bash scripts:

-  Choose a wheel file appropriate for your machine:
  -  CPU:
     https://cntk.ai/pippackages/cpu/cntk-2.0a4-cp34-cp34m-linux\_x86\_64.whl
  -  GPU:
     https://cntk.ai/pippackages/gpu/cntk-2.0a4-cp34-cp34m-linux\_x86\_64.whl

-  Run: ``bash pycntkv2_linux_install.sh <url-of-wheel>``

**Step 3**: Python updates

-  Run:
   ``source $HOME/anaconda3/bin/activate $HOME/anaconda3/envs/cntk-py34``
   to activate the python environment after the environment updates are
   completed.

-  If you want to deploy a new CNTK wheel package at a later time simply
   re-run: ``pip install –upgrade [Whl file location]``

Linux install details
~~~~~~~~~~~~~~~~~~~~~

The script automates the following installation steps:

-  Pre-requisites:
  -  Install for Open MPI
  -  Install for NVIDIA GPU to the latest NVIDIA driver

-  Python:
  -  Install Anaconda (can take a while)
  -  Create a CNTK environment cntk-py34 with conda create
  -  Install CNTK binaries using pip

-  Example / Tutorial:
  -  Git installation
  -  Clone the CNTK repositories. Default directory is
     ``/home/[USERNAME]/repos/cntk``
