//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ConfigFile.cpp : Defines the configuration file loader.
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#include "File.h"
#include "Config.h"
#include "ScriptableObjects.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// ParseCommandLine - parse the command line parameters
// argc - count of arguments
// argv - array of argument parameters
// config - config to return
std::string ConfigParameters::ParseCommandLine(int argc, wchar_t* argv[], ConfigParameters& config)
{
    config.SetName(std::string("global"));

    // This vector keeps track of the config files we have already read
    std::vector<std::string> resolvedConfigFiles;
    std::string configString;

    // start at 1, because 0 is the name of the EXE
    for (int i = 1; i < argc; ++i)
    {
        wstring str = argv[i];

        // allow to change current directory, for easier debugging
        wstring cdDescriptor = L"currentDirectory=";
        if (_wcsnicmp(cdDescriptor.c_str(), str.c_str(), cdDescriptor.length()) == 0)
        {
            wstring dir = str.substr(cdDescriptor.length());
            if (_wchdir(dir.c_str()) != 0)
                InvalidArgument("Failed to set the current directory to '%ls'", dir.c_str());
            fprintf(stderr, "Changed current directory to %ls\n", dir.c_str());
        }

        // see if they are loading a config file
        wstring configDescriptor = L"configFile=";
        int compare = _wcsnicmp(configDescriptor.c_str(), str.c_str(), configDescriptor.length());

        // no config file, parse as regular argument
        if (compare)
        {
            configString += (msra::strfun::utf8(str) + "\n");
        }
        else // One or more config file paths specified in a "+"-separated list.
        {
            const std::string filePaths = msra::strfun::utf8(str.substr(configDescriptor.length()));
            std::vector<std::string> filePathsVec = msra::strfun::split(filePaths, "+");
            for (auto filePath : filePathsVec)
            {
                if (std::find(resolvedConfigFiles.begin(), resolvedConfigFiles.end(), filePath) == resolvedConfigFiles.end())
                {
                    // if haven't already read this file, read it
                    resolvedConfigFiles.push_back(filePath);
                    configString += config.ReadConfigFile(filePath);
                    // remember all config directories, for use as include paths by BrainScriptNetworkBuilder
                    GetBrainScriptNetworkBuilderIncludePaths().push_back(File::DirectoryPathOf(msra::strfun::utf16(filePath)));
                }
                else
                    RuntimeError("Cannot specify same config file multiple times at the command line.");
            }
        }
    }
    // now, configString is a concatenation of lines, including parameters from the command line, with comments stripped

    // expand any lines of the form include=
    configString = config.ResolveIncludeStatements(configString, resolvedConfigFiles);

    // convert into a ConfigDictionary--top-level expressions of the form var=val; if val is a block in braces, it is kept verbatim (not parsed inside)
    config.FileParse(configString);
    return configString;
}

// ResolveIncludeStatements - this function takes a config string, and looks for all lines of the
//     form "include=configPaths", where 'configPaths' is a "+" separated list of paths to config files.
//     If it encounters one of these lines, it reads the config files listed in 'configPaths' (in the specified order),
//     and includes the body of each file in the string which is eventually returned by this function.  If the included
//     config file includes other config files, this function will recursively include those files as well.
// configString - the config string within which to look for "include" statements
// resolvedConfigFiles - the paths to all the config files that have already been resolved.  This vector is used to prevent include loops,
//     and to prevent files from being included multiple times.
// returns: The config string, with all the "include" statements replaced with the bodies of the specified config files.
std::string ConfigParser::ResolveIncludeStatements(const std::string& configString, std::vector<std::string>& resolvedConfigFiles)
{
    std::vector<std::string> lines = msra::strfun::split(configString, "\n");
    std::string includeKeyword = "include=";
    std::size_t includeKeywordSize = includeKeyword.size();
    std::string newConfigString;
    for (std::string line : lines)
    {
        if (line.compare(0, includeKeywordSize, includeKeyword) == 0)
        {
            std::string filePaths = line.substr(includeKeywordSize, line.size() - includeKeywordSize);
            if (filePaths.find(openBraceVar) != std::string::npos)
            {
                RuntimeError("Variable usage (eg, \"$varName$\") not supported in \"include\" statements. Explicit path to config file must be provided");
            }

            std::vector<std::string> filePathVec = msra::strfun::split(filePaths, "+");
            for (auto filePath : filePathVec)
            {
                // if file hasn't already been resolved (the resolvedPaths vector doesn't contain it), resolve it.
                if (std::find(resolvedConfigFiles.begin(), resolvedConfigFiles.end(), filePath) == resolvedConfigFiles.end())
                {
                    // Recursively resolve the include statements in the included config files.
                    // Ensure that the same config file isn't included twice, by keeping track of the config
                    // files that have already been resolved in the resolvedPaths vector.
                    resolvedConfigFiles.push_back(filePath);
                    newConfigString += ResolveIncludeStatements(ReadConfigFile(filePath), resolvedConfigFiles);
                }
                else
                {
                    // We already resolved this path.  Write a warning so that user is aware of this.
                    // TODO: This message is written to stderr before stderr gets redirected to the specified file.  Fix this.
                    fprintf(stderr, "Warning: Config file included multiple times.  Not including config file again: %s", filePath.c_str());
                }
            }
        }
        else
        {
            newConfigString += (line + "\n");
        }
    }
    return newConfigString;
}

// LoadConfigFiles - load multiple configuration file, and adds to config parameters
// filePaths - A "+" delimited list of file paths, corresponding to config files to load
// configStringToAppend - A config string which should be processed together with the config files
void ConfigParser::LoadConfigFiles(const std::wstring& filePaths, const std::string* configStringToAppend)
{
    std::string configString = ReadConfigFiles(filePaths);
    if (configStringToAppend != nullptr)
    {
        configString += *configStringToAppend;
    }

    FileParse(configString);
}

// LoadConfigFileAndResolveVariables - load a configuration file, and add to config parameters.
//     If the config file contains references to variables, which are defined in the 'config' ConfigParameters,
//     then this method will resolve those variables.  This method is meant for the processing of NDL/MEL config files,
//     in order to allow them to access variables defined in the primary config file via $varName$ syntax.
// filePath - filePath to the file to load
// config - These ConfigParameters are used in order to resolve the $varName$ instances in the config file.
void ConfigParser::LoadConfigFileAndResolveVariables(const std::wstring& filePath, const ConfigParameters& config)
{
    // read file, resolve variables, and then parse.
    std::string fileContents = ReadConfigFile(filePath);
    fileContents = config.ResolveVariables(fileContents);
    FileParse(fileContents);
}

// LoadConfigFile - load a configuration file, and add to config parameters
// filePath - filePath to the file to read
void ConfigParser::LoadConfigFile(const std::wstring& filePath)
{
    // read and then parse
    FileParse(ReadConfigFile(filePath));
}

// Same as "ReadConfigFiles" function below, but takes as input string instead of wstring
std::string ConfigParser::ReadConfigFiles(const std::string& filePaths)
{
    return ReadConfigFiles(msra::strfun::utf16(filePaths));
}

// ReadConfigFiles - reads multiple config files, concatenates the content from each file, and returns a string
// filePaths - A "+" delimited list of file paths, corresponding to config files to read
// returns: a string with the concatentated file contents
std::string ConfigParser::ReadConfigFiles(const std::wstring& filePaths)
{
    std::string configString;
    std::vector<std::wstring> filePathVec = msra::strfun::split(filePaths, L"+");
    for (auto filePath : filePathVec)
    {
        configString += ReadConfigFile(filePath);
    }
    return configString;
}

// Same as "ReadConfigFile" function below, but takes as input string instead of wstring
std::string ConfigParser::ReadConfigFile(const std::string& filePath)
{
    return ReadConfigFile(msra::strfun::utf16(filePath));
}

// ReadConfigFile - read a configuration file, and return all lines, stripped of comments, concatenated by newlines, as one long string (no other processing, expansion etc.)
// filePath - the path to the config file to read
// returns: a string with the concatentated file contents
std::string ConfigParser::ReadConfigFile(const std::wstring& filePath)
{
    File file(filePath, fileOptionsRead);

    // initialize configName with file name
    std::string configName = msra::strfun::utf8(filePath);
    auto location = configName.find_last_of("/\\");
    if (location != npos)
        configName = configName.substr(location + 1);
    m_configName = move(configName);

    // read the entire file into a string
    // CONSIDER: should the File API support this, instead of us having to call it line by line?
    size_t fileLength = file.CanSeek() ? file.Size() : 0;
    string str;
    string configFile;
    configFile.reserve(fileLength);
    while (!file.IsEOF())
    {
        file.GetLine(str);
        str = StripComments(str);
        if (str != "")
        {
            configFile.append(str);
            configFile.append("\n");
        }
    }
    return configFile;
}

// GetFileConfigNames - determine the names of the features and labels sections in the config file
// features - [in,out] a vector of feature name strings
// labels - [in,out] a vector of label name strings
template <class ConfigRecordType>
void GetFileConfigNames(const ConfigRecordType& config, std::vector<std::wstring>& features, std::vector<std::wstring>& labels)
{
    for (const auto& id : config.GetMemberIds())
    {
        if (!config.CanBeConfigRecord(id))
            continue;
        const ConfigRecordType& temp = config(id);
        // ############### BREAKING ############
        // Before it required a "dim" parameter, but that was unused for labels.
        // ############### BREAKING ############
        //// see if we have a config parameters that contains a "dim" element, it's a sub key, use it
        //if (temp.ExistsCurrent(L"dim"))
        //{
        // any sub-dictionary that contains any relevant entries is considered an input stream, either label or features
        if (temp.ExistsCurrent(L"labelMappingFile") || temp.ExistsCurrent(L"labelDim") || temp.ExistsCurrent(L"labelType") || (temp.ExistsCurrent(L"sectionType") && (const wstring&) temp(L"sectionType") == L"labels"))
            labels.push_back(id);
        else if (temp.ExistsCurrent(L"dim"))
            features.push_back(id);
        //}
    }
}

template void GetFileConfigNames<ConfigParameters>(const ConfigParameters&, std::vector<std::wstring>& features, std::vector<std::wstring>& labels);
template void GetFileConfigNames<ScriptableObjects::IConfigRecord>(const ScriptableObjects::IConfigRecord&, std::vector<std::wstring>& features, std::vector<std::wstring>& labels);

// FindConfigNames - determine the names of the heirarchy of sections in the config file that contain a particular key
// config - configuration to search
// key - string we ar searching for in each config section
// names - [in,out] a vector of section names in "path" format (i.e. base\subsection)
template <class ConfigRecordType>
void FindConfigNames(const ConfigRecordType& config, std::string key, std::vector<std::wstring>& names)
{
    wstring wkey = wstring(key.begin(), key.end());
    for (const auto& id : config.GetMemberIds())
    {
        if (!config.CanBeConfigRecord(id))
            continue;
        const ConfigRecordType& temp = config(id);
        // see if we have a config parameters that contains a "key" element, if so use it
        if (temp.ExistsCurrent(wkey.c_str()))
        {
            names.push_back(id);
        }
    }
}

template void FindConfigNames<ConfigParameters>(const ConfigParameters&, std::string key, std::vector<std::wstring>& names);
template void FindConfigNames<ScriptableObjects::IConfigRecord>(const ScriptableObjects::IConfigRecord&, std::string key, std::vector<std::wstring>& names);

// Trim - trim white space off the start and end of the string
// str - string to trim
// NOTE: if the entire string is empty, then the string will be set to an empty string
void Trim(std::string& str)
{
    auto found = str.find_first_not_of(" \t");
    if (found == npos)
    {
        str.erase(0);
        return;
    }
    str.erase(0, found);
    found = str.find_last_not_of(" \t");
    if (found != npos)
        str.erase(found + 1);
}

// TrimQuotes - trim surrounding quotation marks
// str - string to trim
void TrimQuotes(std::string& str)
{
    if (str.empty())
        return;
    if (str.front() == '"' && str.back() == '"')
        str = str.substr(1, str.size() - 2);
}

}}}
