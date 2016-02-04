#!/usr/bin/python

# run the Char-RNN training 
#
# TrainCharRNN.py
#
# trying to make this somewhat independent of Python 2 v Python 3 and Unix v Windows
# intended to simplify running workflow instead of typing in long paths
# so far this works when run from Visual Studio Python Tools, Windows command.exe, and MingW64

from __future__ import print_function;
import sys, os, string;

def __main__():

    CurrentDirectory = os.getcwd(); # expect this to be ./CNTK/Examples/Text/Char-RNN
    ExecutablePath = "../../../x64/Debug/";
    DataPath = "./Data/";
    OutputPath = "./Models/";
    ConfigPath = "./Config/";
    OSPathStyle = None;

    # what's the shell path and likely operating system?
    comspec = os.environ.get('COMSPEC');    # test for Windows style environment
    if comspec is None: 
        shell = os.environ.get('SHELL');    # test for Unix style environment
        if shell is None:
            print ("Can't identify path to command shell\n");
        else:
            print ("Unix-style environment detected\n");
            OSPathStyle = "Unix";
    else:
        shell = comspec; 
        print ("Windows-style environment detected\n");
        OSPathStyle = "Windows";

    # Construct the command line and swap delimiters if needed
    CmdString = ExecutablePath + "CNTK.exe" + " " \
        + "configFile" + "=" + ConfigPath + "Char-RNN.cntk";
    if OSPathStyle == "Windows":
        CmdString = SwapToWindowsPathStyle(CmdString);
        ExecutablePath = SwapToWindowsPathStyle(ExecutablePath);
        DataPath = SwapToWindowsPathStyle(DataPath);
        ConfigPath = SwapToWindowsPathStyle(ConfigPath);

    print ("cwd = " + CurrentDirectory + "\n");
    print ("CmdString = " + CmdString + "\n");

    os.system(CmdString);


#   pass in a Unix style path, return a Windows-style path, swapping "/" and "\"
def SwapToWindowsPathStyle(pathstring):
        return str.replace(pathstring, "/", "\\");

__main__();
         
