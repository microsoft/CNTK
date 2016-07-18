//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTK.cpp : Defines the entry point for the console application.
//

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _

#include "stdafx.h"
#include "Basics.h"
#include "Actions.h"
#include "ComputationNetwork.h"
#include "ComputationNode.h"
#include "DataReader.h"
#include "DataWriter.h"
#include "SimpleNetworkBuilder.h"
#include "NDLNetworkBuilder.h"
#include "ModelEditLanguage.h"
#include "CPUMatrix.h" // used for SetNumThreads()
#include "CommonMatrix.h"
#include "SGD.h"
#include "MPIWrapper.h"
#include "Config.h"
#include "SimpleEvaluator.h"
#include "SimpleOutputWriter.h"
#include "BestGpu.h"
#include "ProgressTracing.h"
#include "fileutil.h"
#include "ScriptableObjects.h"
#include "BrainScriptEvaluator.h"
#include "BrainScriptParser.h"

#include <string>
#include <chrono>
#include <algorithm>
#if defined(_WIN32)
#include "io.h"
#include <DelayImp.h>
#pragma comment(lib, "Delayimp.lib")
#pragma comment(lib, "shlwapi.lib")
#endif
#include "buildinfo.h"
#include "hostname.h"
#ifdef LEAKDETECT
#include "vld.h" // for memory leak detection
#endif
#include <vector>
#include <iostream>
#include <queue>
#include <set>
#include <memory>

#ifndef let
#define let const auto
#endif

// TODO: Temporary mechanism to enable memory sharing for
// node output value matrices. This will go away when the
// sharing is ready to be enabled by default
bool g_shareNodeValueMatrices = false;

using namespace std;
using namespace Microsoft::MSR;
using namespace Microsoft::MSR::CNTK;

// internal test routine forward declaration
template <typename ElemType>
void TestCn(const ConfigParameters& config);

void RedirectStdErr(wstring logpath)
{
    LOGPRINTF(stderr, "Redirecting stderr to file %S\n", logpath.c_str());
    auto f = make_shared<File>(logpath.c_str(), fileOptionsWrite | fileOptionsText);
    if (dup2(fileno(*f), 2) == -1)
    {
        RuntimeError("unexpected failure to redirect stderr to log file");
    }
    setvbuf(stderr, NULL, _IONBF, 16384); // unbuffer it
    static auto fKept = f;                // keep it around (until it gets changed)
}

std::string WCharToString(const wchar_t* wst)
{
    std::wstring ws(wst);
    std::string s(ws.begin(), ws.end());
    s.assign(ws.begin(), ws.end());
    return s;
}

// TODO: This is an action, it should be moved into ActionsLib.
template <typename ElemType>
void DumpNodeInfo(const ConfigParameters& config)
{
    wstring modelPath = config(L"modelPath");
    wstring nodeName = config(L"nodeName", L"__AllNodes__");
    wstring nodeNameRegexStr = config(L"nodeNameRegex", L"");
    wstring defOutFilePath = modelPath + L"." + nodeName + L".txt";
    wstring outputFile = config(L"outputFile", defOutFilePath);
    bool printValues = config(L"printValues", true);
    bool printMetadata = config(L"printMetadata", true);
    if (!printValues && !printMetadata)
    {
        InvalidArgument("printValues and printMetadata: Since both are set to false, there will be nothing to dump");
    }

    ComputationNetwork net(-1);    // always use CPU
    net.Load<ElemType>(modelPath); // TODO: we have a function now to combine this and the previous line
    net.DumpNodeInfoToFile(nodeName, printValues, printMetadata, outputFile, nodeNameRegexStr);
}

size_t GetMaxEpochs(const ConfigParameters& configParams)
{
    ConfigParameters configSGD(configParams("SGD"));
    size_t maxEpochs = configSGD("maxEpochs");

    return maxEpochs;
}

#ifndef CPUONLY
// abort execution is GPU is not supported (e.g. compute capability not supported)
void CheckSupportForGpu(DEVICEID_TYPE deviceId)
{
    auto gpuData = GetGpuData(deviceId);
    if (gpuData.validity == GpuValidity::ComputeCapabilityNotSupported)
    {
        InvalidArgument("CNTK: The GPU (%s) has compute capability %d.%d.  CNTK is only supported on GPUs with compute capability 3.0 or greater", 
                        gpuData.name.c_str(), gpuData.versionMajor, gpuData.versionMinor);
    }
    else if (gpuData.validity == GpuValidity::UnknownDevice)
    {
        InvalidArgument("CNTK: Unknown GPU with Device ID %d.", gpuData.deviceId);
    }
}
#endif

// special temporary function to guard against a now invalid usage of "truncated" which exists in some IPG production setups
static void DisableLegacyTruncationSettings(const ConfigParameters& TopLevelConfig, const ConfigParameters& commandConfig)
{
    if (TopLevelConfig.ExistsCurrent(L"Truncated"))
    {
        return;
    }

    // if any of the action has set a reader/SGD section and has different Truncated value for reader and SGD section
    ConfigArray actions = commandConfig(L"action");
    for (size_t i = 0; i < actions.size(); i++)
    {
        if (actions[i] == "train" || actions[i] == "trainRNN")
        {
            ConfigParameters sgd = ConfigParameters(commandConfig(L"SGD"));
            ConfigParameters reader = ConfigParameters(commandConfig(L"reader"));
            // reader and SGD sections are two must-have sections in train/trainRNN
            if (reader.ExistsCurrent(L"Truncated") && !sgd.ExistsCurrent(L"Truncated"))
            {
                InvalidArgument("DisableLegacyUsage: setting Truncated only in reader section are not allowed. Please move Truncated=true/false to the top level section.");
            }
        }
    }
}
static void DisableLegacyUsage(const ConfigParameters& TopLevelConfig, const ConfigArray& commands)
{
    for (size_t i = 0; i < commands.size(); i++)
    {
        ConfigParameters cfgParameters(TopLevelConfig(commands[i]));
        DisableLegacyTruncationSettings(TopLevelConfig, cfgParameters);
    }
}

// When running in parallel with MPI, only commands in 'commandstoRunOnAllRanks' should
// be run in parallel across multiple ranks. Others should only run on rank 0
const std::set<std::string> commandstoRunOnAllRanks = { "train", "trainRNN", "adapt", "test", "eval", "cv", "devtest" };

// process the command
template <typename ElemType>
void DoCommands(const ConfigParameters& config, const shared_ptr<MPIWrapper>& mpi)
{
    ConfigArray command = config(L"command", "train");

    int numCPUThreads = config(L"numCPUThreads", "0");
    numCPUThreads = CPUMatrix<ElemType>::SetNumThreads(numCPUThreads);

    if (numCPUThreads > 0)
    {
        LOGPRINTF(stderr, "Using %d CPU threads.\n", numCPUThreads);
    }

    bool progressTracing = config(L"progressTracing", false);

    // temporary hack to prevent users from failing due to a small breaking change related to the "truncated" flag (will be redone bigger and better some day)
    DisableLegacyUsage(config, command);

    // summarize command info upfront in the log and stdout
    size_t fullTotalMaxEpochs = 0;
    for (int i = 0; i < command.size(); i++)
    {
        // get the configuration parameters that match the command
        ConfigParameters commandParams(config(command[i]));
        ConfigArray action = commandParams("action", "train");

        // determine the action to perform, and do it
        for (int j = 0; j < action.size(); j++)
        {
            if (action[j] == "train" || action[j] == "trainRNN")
            {
                wstring modelPath = commandParams("modelPath");
                LOGPRINTF(stderr, "CNTKModelPath: %ls\n", modelPath.c_str());
                size_t maxEpochs = GetMaxEpochs(commandParams);
                LOGPRINTF(stderr, "CNTKCommandTrainInfo: %s : %d\n", command[i].c_str(), (int) maxEpochs);
                fullTotalMaxEpochs += maxEpochs;
            }
        }
    }
    LOGPRINTF(stderr, "CNTKCommandTrainInfo: CNTKNoMoreCommands_Total : %d\n", (int) fullTotalMaxEpochs);

    // set up progress tracing for compute cluster management
    if (progressTracing && (!mpi || mpi->IsMainNode()))
    {
        ProgressTracing::SetTracingFlag();
        ProgressTracing::TraceTotalNumberOfSteps(fullTotalMaxEpochs); // enable tracing, using this as the total number of epochs
    }

    size_t fullEpochsOffset = 0;

    // execute the commands
    for (int i = 0; i < command.size(); i++)
    {
        // get the configuration parameters that match the command
        ConfigParameters commandParams(config(command[i]));
        ConfigArray action = commandParams("action", "train");

        if (progressTracing && ((mpi == nullptr) || mpi->IsMainNode()))
        {
            ProgressTracing::SetStepOffset(fullEpochsOffset); // this is the epoch number that SGD will log relative to
        }

        // determine the action to perform, and do it
        for (int j = 0; j < action.size(); j++)
        {
            string thisAction = action[j];

            // print a banner to visually separate each action in the log
            const char* delim = "##############################################################################";
            const char* prefix = "Action ";
            fprintf(stderr, "\n");
            LOGPRINTF(stderr, "%s\n", delim);
            LOGPRINTF(stderr, "#%*s#\n", (int)(strlen(delim) - 2), "");
            LOGPRINTF(stderr, "# %s\"%s\"%*s #\n", prefix, thisAction.c_str(), (int)(strlen(delim) - strlen(prefix) - thisAction.size() - 6), "");
            LOGPRINTF(stderr, "#%*s#\n", (int)(strlen(delim) - 2), "");
            LOGPRINTF(stderr, "%s\n\n", delim);

            if ((mpi == nullptr) || (commandstoRunOnAllRanks.find(thisAction) != commandstoRunOnAllRanks.end()) || mpi->IsMainNode())
            {
                if (thisAction == "train" || thisAction == "trainRNN")
                {
                    LOGPRINTF(stderr, "CNTKCommandTrainBegin: %s\n", command[i].c_str());
                    DoTrain<ConfigParameters, ElemType>(commandParams);
                    LOGPRINTF(stderr, "CNTKCommandTrainEnd: %s\n", command[i].c_str());
                    fullEpochsOffset += GetMaxEpochs(commandParams);
                }
                else if (thisAction == "adapt")
                {
                    DoAdapt<ElemType>(commandParams);
                }
                else if (thisAction == "test" || thisAction == "eval")
                {
                    DoEval<ElemType>(commandParams);
                }
                else if (thisAction == "edit")
                {
                    DoEdit<ElemType>(commandParams);
                }
                else if (thisAction == "cv")
                {
                    DoCrossValidate<ElemType>(commandParams);
                }
                else if (thisAction == "write")
                {
                    DoWriteOutput<ElemType>(commandParams);
                }
                else if (thisAction == "devtest")
                {
                    TestCn<ElemType>(config); // for "devtest" action pass the root config instead
                }
                else if (thisAction == "dumpNode" /*deprecated:*/|| thisAction == "dumpnode")
                {
                    DumpNodeInfo<ElemType>(commandParams);
                }
                else if (thisAction == "convertdbn")
                {
                    DoConvertFromDbn<ElemType>(commandParams);
                }
                else if (thisAction == "exportdbn")
                {
                    DoExportToDbn<ElemType>(commandParams);
                }
                else if (thisAction == "createLabelMap")
                {
                    DoCreateLabelMap<ElemType>(commandParams);
                }
                else if (thisAction == "writeWordAndClass")
                {
                    DoWriteWordAndClassInfo<ElemType>(commandParams);
                }
                else if (thisAction == "plot")
                {
                    DoTopologyPlot<ElemType>(commandParams);
                }
                else if (thisAction == "SVD")
                {
                    DoParameterSVD<ElemType>(commandParams);
                }
                else
                {
                    RuntimeError("unknown action: %s  in command set: %s", thisAction.c_str(), command[i].c_str());
                }
            }

            fprintf(stderr, "\n");
            LOGPRINTF(stderr, "Action \"%s\" complete.\n\n", thisAction.c_str());

            NDLScript<ElemType> ndlScript;
            ndlScript.ClearGlobal(); // clear global macros between commands

            // Synchronize all ranks before proceeding to next action/command
            if (mpi)
                mpi->WaitAll();
        }
    }
}

std::string TimeDateStamp()
{
    time_t t = time(NULL);
    struct tm now = *localtime(&t);
    char buf[30];
    sprintf(buf, "%04d/%02d/%02d %02d:%02d:%02d", now.tm_year + 1900, now.tm_mon + 1, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec);
    return buf;
}

void PrintBuiltInfo()
{
    LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
    LOGPRINTF(stderr, "Build info: \n\n");
    LOGPRINTF(stderr, "\t\tBuilt time: %s %s\n", __DATE__, __TIME__);
    LOGPRINTF(stderr, "\t\tLast modified date: %s\n", __TIMESTAMP__);
#ifdef _BUILDTYPE_
    LOGPRINTF(stderr, "\t\tBuild type: %s\n", _BUILDTYPE_);
#endif
#ifdef _BUILDTARGET_
    LOGPRINTF(stderr, "\t\tBuild target: %s\n", _BUILDTARGET_);
#endif
#ifdef _WITH_1BITSGD_
    LOGPRINTF(stderr, "\t\tWith 1bit-SGD: %s\n", _WITH_1BITSGD_);
#endif
#ifdef _MATHLIB_
    LOGPRINTF(stderr, "\t\tMath lib: %s\n", _MATHLIB_);
#endif
#ifdef _CUDA_PATH_
    LOGPRINTF(stderr, "\t\tCUDA_PATH: %s\n", _CUDA_PATH_);
#endif
#ifdef _CUB_PATH_
    LOGPRINTF(stderr, "\t\tCUB_PATH: %s\n", _CUB_PATH_);
#endif
#ifdef _CUDNN_PATH_
    LOGPRINTF(stderr, "\t\tCUDNN_PATH: %s\n", _CUDNN_PATH_);
#endif
#ifdef _GIT_EXIST
    LOGPRINTF(stderr, "\t\tBuild Branch: %s\n", _BUILDBRANCH_);
    LOGPRINTF(stderr, "\t\tBuild SHA1: %s\n", _BUILDSHA1_);
#endif
#ifdef _BUILDER_
    LOGPRINTF(stderr, "\t\tBuilt by %s on %s\n", _BUILDER_, _BUILDMACHINE_);
#endif
#ifdef _BUILDPATH_
    LOGPRINTF(stderr, "\t\tBuild Path: %s\n", _BUILDPATH_);
#endif
    LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
}

void PrintUsageInfo()
{
    LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
    LOGPRINTF(stderr, "Usage: cntk configFile=yourConfigFile\n");
    LOGPRINTF(stderr, "For detailed information please consult the CNTK book\n");
    LOGPRINTF(stderr, "\"An Introduction to Computational Networks and the Computational Network Toolkit\"\n");
    LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
}

// print gpu info for current gpu devices (e.g. Device[0]: cores = 2496; computeCapability = 5.2; type = "Quadro M4000"; memory = 8192 MB)
void PrintGpuInfo()
{
#ifndef CPUONLY
    std::vector<GpuData> gpusData = GetAllGpusData();

    if (gpusData.empty())
    {
        LOGPRINTF(stderr, "No GPUs found\n");
        return;
    }

    LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
    LOGPRINTF(stderr, "GPU info:\n\n");

    for (GpuData& data : gpusData)
    {
        LOGPRINTF(stderr, "\t\tDevice[%d]: cores = %d; computeCapability = %d.%d; type = \"%s\"; memory = %lu MB\n",
                  data.deviceId, data.cudaCores, data.versionMajor, data.versionMinor, data.name.c_str(), data.totalMemory);
    }
    LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
#endif
}

// ---------------------------------------------------------------------------
// main() for use with BrainScript
// ---------------------------------------------------------------------------

wstring ConsumeArg(vector<wstring>& args)
{
    if (args.empty())
        InvalidArgument("Unexpected end of command line.");
    wstring arg = args.front();
    args.erase(args.begin());
    return arg;
}
template <class WHAT>
static void Append(vector<wstring>& toWhat, const WHAT& what)
{
    toWhat.insert(toWhat.end(), what.begin(), what.end());
}
static wstring PathToBSStringLiteral(const wstring& path) // quote a pathname for BS
{
    let hasSingleQuote = path.find(path, L'\'') != wstring::npos;
    let hasDoubleQuote = path.find(path, L'"') != wstring::npos;
    if (hasSingleQuote && hasDoubleQuote)
        InvalidArgument("Pathname cannot contain both single (') and double (\") quote at the same time: %ls", path.c_str());
    else if (hasSingleQuote)
        return L"\"" + path + L"\"";
    else
        return L'"' + path + L'"';
}

// TODO: decide where these should go. Also, do we need three variables?
//extern wstring standardFunctions;
//extern wstring commonMacros;
//extern wstring computationNodes;

int wmainWithBS(int argc, wchar_t* argv[]) // called from wmain which is a wrapper that catches & reports Win32 exceptions
{
    vector<wstring> args(argv, argv + argc);
    let exePath = ConsumeArg(args);

    // startup message
    // In case of a redirect of stderr, this will be printed twice, once upfront, and once again after the redirect so that it goes into the log file
    wstring startupMessage = msra::strfun::wstrprintf(L"running on %ls at %ls\n", msra::strfun::utf16(GetHostName()).c_str(), msra::strfun::utf16(TimeDateStamp()).c_str());
    startupMessage += msra::strfun::wstrprintf(L"command line: %ls", exePath.c_str());
    for (const auto& arg : args)
        startupMessage += L"  " + arg;

    LOGPRINTF(stderr, "%ls\n", startupMessage.c_str());

    // parse command-line options
    vector<wstring> sourceFiles;
    vector<wstring> includePaths;
    vector<wstring> overrides;
    wstring workingDir;
    while (!args.empty())
    {
        let option = ConsumeArg(args);
        if (option == L"-f" || option == L"--file") // -f defines source files
            Append(sourceFiles, msra::strfun::split(ConsumeArg(args), L";"));
        else if (option == L"-I") // -I declares an include search path
            Append(includePaths, msra::strfun::split(ConsumeArg(args), L";"));
        else if (option == L"-D") // -D defines variables inline on the command line (which may override BS)
            overrides.push_back(ConsumeArg(args));
        else if (option == L"--cd") // --cd sets the working directory
            workingDir = ConsumeArg(args);
        else
            InvalidArgument("Invalid command-line option '%ls'.", option.c_str());
    }

    // change working directory
    if (workingDir != L"")
        _wchdir(workingDir.c_str());

    // compile the BrainScript
    wstring bs = L"[\n";
    bs += L"include \'cntk.core.bs'"; // start with including the standard macros

    // Note: Using lowercase ^^ here to match the Linux name of the CNTK exe.
    //bs += standardFunctions + computationNodes + commonMacros + L"\n";
    for (const auto& sourceFile : sourceFiles)
        bs += L"include " + PathToBSStringLiteral(sourceFile) + L"\n";
    bs += L"\n]\n";
    for (const auto& over : overrides)
        bs += L"with [ " + over + L" ]\n";

    fprintf(stderr, "\n\n");
    LOGPRINTF(stderr, "BrainScript -->\n\n%ls\n\n", bs.c_str());

    let expr = BS::ParseConfigExpression(bs, move(includePaths)); // parse
    let valp = BS::Evaluate(expr);                                // evaluate parse into a dictionary
    let& config = valp.AsRef<ScriptableObjects::IConfigRecord>(); // this is the dictionary

#ifndef CPUONLY
    auto valpp = config.Find(L"deviceId");
    if (valpp)
    {
        auto valp = *valpp;
        if (!valp.Is<ScriptableObjects::String>()) // if it's not string 'auto' or 'cpu', then it's a gpu
        {
            if (static_cast<int>(valp) >= 0) // gpu (id >= 0)
            {
                CheckSupportForGpu(valp); // throws if gpu is not supported
            }
        }
    }
#endif

    // legacy parameters that have changed spelling
    if (config.Find(L"DoneFile")) // variables follow camel case (start with lower-case letters)
        InvalidArgument("Legacy spelling of 'DoneFile' no longer allowed. Use 'doneFile'.");

    if (config.Find(L"command")) // spelling error, should be plural. Using 'actions' instead to match the data type.
        InvalidArgument("Legacy spelling of 'command' no longer allowed. Use 'actions'.");

    if (config.Find(L"type"))
        InvalidArgument("Legacy name 'type' no longer allowed. Use 'precision'.");

    // parallel training
    shared_ptr<Microsoft::MSR::CNTK::MPIWrapper> mpi;
    auto ensureMPIWrapperCleanup = MakeScopeExit(&MPIWrapper::DeleteInstance);
    bool paralleltrain = config(L"parallelTrain", false);
    if (paralleltrain)
        mpi = MPIWrapper::GetInstance(true /*create*/);

    g_shareNodeValueMatrices = config(L"shareNodeValueMatrices", false);

    TracingGPUMemoryAllocator::SetTraceLevel(config(L"traceGPUMemoryAllocations", 0));

    // logging
    wstring logpath = config(L"stderr", L"");
    if (logpath != L"")
    {
        logpath += L"_actions"; // TODO: for old CNTK, this was a concatenation of all action names, which we no longer know
        logpath += L".log";     // TODO: why do we need to append this here?

        if (paralleltrain)
            logpath += msra::strfun::wstrprintf(L"rank%d", (int) mpi->CurrentNodeRank());

        RedirectStdErr(logpath);
        LOGPRINTF(stderr, "%ls\n", startupMessage.c_str());
    }

    // echo config info to log
    PrintBuiltInfo();

    // echo gpu info to log
    PrintGpuInfo();

    // execute the actions
    // std::string type = config(L"precision", "float");
    int numCPUThreads = config(L"numCPUThreads", 0);
    numCPUThreads = CPUMatrix<float /*any will do*/>::SetNumThreads(numCPUThreads);
    if (numCPUThreads > 0)
        LOGPRINTF(stderr, "Using %d CPU threads.\n", numCPUThreads);

    bool progressTracing = config(L"progressTracing", false);
    size_t fullTotalMaxEpochs = 1; // BUGBUG: BS does not allow me to read out the max epochs parameters, as that would instantiate and thus execute the objects

    // set up progress tracing for compute cluster management
    if (progressTracing && ((mpi == nullptr) || mpi->IsMainNode()))
        ProgressTracing::TraceTotalNumberOfSteps(fullTotalMaxEpochs); // enable tracing, using this as the total number of epochs

    // MAIN LOOP that executes the actions
    auto actionsVal = config[L"actions"];

    // Note: weird behavior. If 'actions' is a scalar value (rather than an array) then it will have been resolved already after the above call. That means, it has already completed its action!
    //       Not pretty, but a direct consequence of the lazy evaluation. The only good solution would be to have a syntax for arrays including length 0 and 1.
    //       Since this in the end behaves indistinguishable from the array loop below, we will keep it for now.
    if (actionsVal.Is<ScriptableObjects::ConfigArray>())
    {
        const ScriptableObjects::ConfigArray& actions = actionsVal;
        for (int i = actions.GetIndexRange().first; i <= actions.GetIndexRange().second; i++)
        {
            // TODO: When running in parallel with MPI, only commands in 'commandstoRunOnAllRanks' should
            // be run in parallel across multiple ranks. Others should only run on rank 0
            actions.At(i, [](const wstring&)
                       {
                       }); // this will evaluate and thus execute the action
        }
    }
    // else action has already been executed, see comment above

    // write a doneFile if requested
    wstring doneFile = config(L"doneFile", L"");
    if (doneFile != L"")
    {
        FILE* fp = fopenOrDie(doneFile.c_str(), L"w");
        fprintf(fp, "successfully finished at %s on %s\n", TimeDateStamp().c_str(), GetHostName().c_str());
        fcloseOrDie(fp);
    }
    // TODO: change this back to COMPLETED, double underscores don't look good in output
    LOGPRINTF(stderr, "__COMPLETED__\n");
    fflush(stderr);

    return EXIT_SUCCESS;
}

// ---------------------------------------------------------------------------
// main() for old CNTK config language
// ---------------------------------------------------------------------------
// called from wmain which is a wrapper that catches & repots Win32 exceptions
int wmainOldCNTKConfig(int argc, wchar_t* argv[])
{
    ConfigParameters config;
    std::string rawConfigString = ConfigParameters::ParseCommandLine(argc, argv, config);    // get the command param set they want

#ifndef CPUONLY
    ConfigValue val = config("deviceId", "auto");
    if (!EqualCI(val, "cpu") && !EqualCI(val, "auto"))
    {
        if (static_cast<int>(val) >= 0) // gpu (id >= 0)
        {
            CheckSupportForGpu(static_cast<int>(val)); // throws if gpu is not supported
        }
    }
#endif

    bool timestamping = config(L"timestamping", false);
    if (timestamping)
    {
        ProgressTracing::SetTimestampingFlag();
    }

    // get the command param set they want
    wstring logpath = config(L"stderr", L"");

    //  [1/26/2015 erw, add done file so that it can be used on HPC]
    wstring DoneFile = config(L"DoneFile", L"");
    ConfigArray command = config(L"command", "train");

    // paralleltrain training
    shared_ptr<Microsoft::MSR::CNTK::MPIWrapper> mpi;
    auto ensureMPIWrapperCleanup = MakeScopeExit(&MPIWrapper::DeleteInstance);
    bool paralleltrain = config(L"parallelTrain", "false");
    if (paralleltrain)
        mpi = MPIWrapper::GetInstance(true /*create*/);

    g_shareNodeValueMatrices = config(L"shareNodeValueMatrices", false);

    TracingGPUMemoryAllocator::SetTraceLevel(config(L"traceGPUMemoryAllocations", 0));

    if (logpath != L"")
    {
        for (int i = 0; i < command.size(); i++)
        {
            logpath += L"_";
            logpath += (wstring) command[i];
        }
        logpath += L".log";

        if (paralleltrain)
        {
            std::wostringstream oss;
            oss << mpi->CurrentNodeRank();
            logpath += L"rank" + oss.str();
        }
        RedirectStdErr(logpath);
    }

    PrintBuiltInfo(); // this one goes to log file
    PrintGpuInfo();

    std::string timestamp = TimeDateStamp();

    // dump config info
    fprintf(stderr, "\n");
    LOGPRINTF(stderr, "Running on %s at %s\n", GetHostName().c_str(), timestamp.c_str());
    LOGPRINTF(stderr, "Command line: \n");
    for (int i = 0; i < argc; i++)
        fprintf(stderr, "%*s%ls", i > 0 ? 2 : 0, "", argv[i]); // use 2 spaces for better visual separability
    fprintf(stderr, "\n\n");

#if 1 //def _DEBUG
    // This simply merges all the different config parameters specified (eg, via config files or via command line directly),
    // and prints it.
    fprintf(stderr, "\n\n");
    LOGPRINTF(stderr, ">>>>>>>>>>>>>>>>>>>> RAW CONFIG (VARIABLES NOT RESOLVED) >>>>>>>>>>>>>>>>>>>>\n");
    LOGPRINTF(stderr, "%s\n", rawConfigString.c_str());
    LOGPRINTF(stderr, "<<<<<<<<<<<<<<<<<<<< RAW CONFIG (VARIABLES NOT RESOLVED)  <<<<<<<<<<<<<<<<<<<<\n");

    // Same as above, but all variables are resolved.  If a parameter is set multiple times (eg, set in config, overridden at command line),
    // All of these assignments will appear, even though only the last assignment matters.
    fprintf(stderr, "\n");
    LOGPRINTF(stderr, ">>>>>>>>>>>>>>>>>>>> RAW CONFIG WITH ALL VARIABLES RESOLVED >>>>>>>>>>>>>>>>>>>>\n");
    LOGPRINTF(stderr, "%s\n", config.ResolveVariables(rawConfigString).c_str());
    LOGPRINTF(stderr, "<<<<<<<<<<<<<<<<<<<< RAW CONFIG WITH ALL VARIABLES RESOLVED <<<<<<<<<<<<<<<<<<<<\n");

    // This outputs the final value each variable/parameter is assigned to in config (so if a parameter is set multiple times, only the last
    // value it is set to will appear).
    fprintf(stderr, "\n");
    LOGPRINTF(stderr, ">>>>>>>>>>>>>>>>>>>> PROCESSED CONFIG WITH ALL VARIABLES RESOLVED >>>>>>>>>>>>>>>>>>>>\n");
    config.dumpWithResolvedVariables();
    LOGPRINTF(stderr, "<<<<<<<<<<<<<<<<<<<< PROCESSED CONFIG WITH ALL VARIABLES RESOLVED <<<<<<<<<<<<<<<<<<<<\n");
#endif

    LOGPRINTF(stderr, "Commands:");
    for (int i = 0; i < command.size(); i++)
        fprintf(stderr, " %s", command[i].c_str());
    fprintf(stderr, "\n");

    // run commands
    std::string type = config(L"precision", "float");
    // accept old precision key for backward compatibility
    if (config.Exists("type"))
        InvalidArgument("CNTK: Use of 'type' parameter is deprecated, it is called 'precision' now.");

    LOGPRINTF(stderr, "Precision = \"%s\"\n", type.c_str());

    if (type == "float")
        DoCommands<float>(config, mpi);
    else if (type == "double")
        DoCommands<double>(config, mpi);
    else
        RuntimeError("CNTK: Invalid precision string: \"%s\", must be \"float\" or \"double\"", type.c_str());

    // if completed then write a DoneFile if requested
    if (!DoneFile.empty())
    {
        FILE* fp = fopenOrDie(DoneFile.c_str(), L"w");
        fprintf(fp, "successfully finished at %s on %s\n", TimeDateStamp().c_str(), GetHostName().c_str());
        fcloseOrDie(fp);
    }
    // TODO: Change back to COMPLETED (no underscores)
    LOGPRINTF(stderr, "__COMPLETED__\n");
    fflush(stderr);

    return EXIT_SUCCESS;
}

// new_handler to print call stack upon allocation failure
void AllocationFailureHandler()
{
    Microsoft::MSR::CNTK::DebugUtil::PrintCallStack();
    std::set_new_handler(nullptr);
    throw std::bad_alloc();
}

// ---------------------------------------------------------------------------
// main wrapper that catches C++ exceptions and prints them
// ---------------------------------------------------------------------------

int wmain1(int argc, wchar_t* argv[]) // called from wmain which is a wrapper that catches & reports Win32 exceptions
{
    std::set_new_handler(AllocationFailureHandler);

    try
    {        
        PrintBuiltInfo(); // print build info directly in case that user provides zero argument (convenient for checking build type)

        if (argc <= 1)
        {
            LOGPRINTF(stderr, "No command-line argument given.\n");
            PrintUsageInfo();
            return EXIT_FAILURE;
        }

        // detect legacy CNTK configuration
        bool isOldCNTKConfig = false;
        for (int i = 0; i < argc && !isOldCNTKConfig; i++)
            isOldCNTKConfig |= !_wcsnicmp(L"configFile=", argv[i], 11);

        if (isOldCNTKConfig)
            return wmainOldCNTKConfig(argc, argv);

        // run from BrainScript
        return wmainWithBS(argc, argv);
    }
    catch (const ScriptableObjects::ScriptingException& err)
    {
        fprintf(stderr, "\n");
        err.PrintError(ProgressTracing::GetTimeStampPrefix() + L"EXCEPTION occurred");
        return EXIT_FAILURE;
    }
    catch (const IExceptionWithCallStackBase& err)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s", err.CallStack());
        LOGPRINTF(stderr, "EXCEPTION occurred: %s\n", dynamic_cast<const std::exception&>(err).what());
        return EXIT_FAILURE;
    }
    catch (const std::exception& err)
    {
        fprintf(stderr, "\n");
        LOGPRINTF(stderr, "EXCEPTION occurred: %s\n", err.what());
        return EXIT_FAILURE;
    }
    catch (...)
    {
        fprintf(stderr, "\n");
        LOGPRINTF(stderr, "Unknown ERROR occurred\n");
        return EXIT_FAILURE;
    }
}

#ifdef __WINDOWS__
void TerminateThis()
{
    LOGPRINTF(stderr, "terminate_this: aborting\n");
    fflush(stderr);
    exit(EXIT_FAILURE);
}

#define EXCEPTION_DLL_NOT_FOUND VcppException(ERROR_SEVERITY_ERROR, ERROR_MOD_NOT_FOUND)

static void LogDelayLoadError(PEXCEPTION_POINTERS pExcPointers)
{
    if (pExcPointers->ExceptionRecord->ExceptionCode == EXCEPTION_DLL_NOT_FOUND)
    {
        const auto & pDelayLoadInfo = *PDelayLoadInfo(pExcPointers->ExceptionRecord->ExceptionInformation[0]);
        LOGPRINTF(stderr, "CNTK: Failed to load DLL '%s'.\n", pDelayLoadInfo.szDll);
    }
}

int wmain(int argc, wchar_t* argv[]) // wmain wrapper that reports Win32 exceptions
{
    set_terminate(TerminateThis);    // insert a termination handler to ensure stderr gets flushed before actually terminating
    _set_error_mode(_OUT_TO_STDERR); // make sure there are no CRT prompts when CNTK is executing

    // Note: this does not seem to work--processes with this seem to just hang instead of terminating
    __try
    {
        return wmain1(argc, argv);
    }
    __except (LogDelayLoadError(GetExceptionInformation()), EXCEPTION_EXECUTE_HANDLER)
    {
        auto code = GetExceptionCode();
        const char * msg = "";
        if      (code == EXCEPTION_ACCESS_VIOLATION)   msg = ": Access violation"; // the famous 0xc0000005 error
        else if (code == EXCEPTION_INT_DIVIDE_BY_ZERO) msg = ": Integer division by zero";
        else if (code == EXCEPTION_STACK_OVERFLOW)     msg = ": Stack overflow";
        else if (code == EXCEPTION_DLL_NOT_FOUND)      msg = ": Module not found";
        LOGPRINTF(stderr, "CNTK: Caught Win32 exception 0x%08x%s.\n", (unsigned int)code, msg);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}
#endif

#ifdef __UNIX__
/// UNIX main function converts arguments in UTF-8 encoding and passes to Visual-Studio style wmain() which takes wchar_t strings.
int main(int argc, char* argv[])
{
    // TODO: change to STL containers
    wchar_t** wargs = new wchar_t*[argc];
    for (int i = 0; i < argc; ++i)
    {
        wargs[i] = new wchar_t[strlen(argv[i]) + 1];
        size_t ans = ::mbstowcs(wargs[i], argv[i], strlen(argv[i]) + 1);
        assert(ans == strlen(argv[i]));
    }
    int ret = wmain1(argc, wargs);
    for (int i = 0; i < argc; ++i)
        delete[] wargs[i];
    delete[] wargs;
    return ret;
}
#endif
