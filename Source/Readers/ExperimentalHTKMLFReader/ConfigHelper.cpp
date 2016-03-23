//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <regex>
#include "ConfigHelper.h"
#include "DataReader.h"
#include "StringUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

pair<size_t, size_t> ConfigHelper::GetContextWindow()
{
    size_t left = 0, right = 0;
    intargvector contextWindow = m_config(L"contextWindow", ConfigParameters::Array(intargvector(vector<int>{1})));

    if (contextWindow.size() == 1) // symmetric
    {
        size_t windowFrames = contextWindow[0];
        if (windowFrames % 2 == 0)
        {
            InvalidArgument("Neighbor expansion of input features to %d is not symmetrical.", (int)windowFrames);
        }

        // extend each side by this
        size_t context = windowFrames / 2;
        left = context;
        right = context;
    }
    else if (contextWindow.size() == 2)
    {
        // left context, right context
        left = contextWindow[0];
        right = contextWindow[1];
    }
    else
    {
        InvalidArgument("contextWindow must have 1 or 2 values specified, found %d.", (int)contextWindow.size());
    }

    return make_pair(left, right);
}

void ConfigHelper::CheckFeatureType()
{
    wstring type = m_config(L"type", L"real");
    if (_wcsicmp(type.c_str(), L"real"))
    {
        InvalidArgument("Feature type must be of type 'real'.");
    }
}

void ConfigHelper::CheckLabelType()
{
    wstring type;
    if (m_config.Exists(L"labelType"))
    {
        // TODO: let's deprecate this eventually and just use "type"...
        type = static_cast<const wstring&>(m_config(L"labelType"));
    }
    else
    {
        // outputs should default to category
        type = static_cast<const wstring&>(m_config(L"type", L"category"));
    }

    if (_wcsicmp(type.c_str(), L"category"))
    {
        InvalidArgument("Label type must be of type 'category'.");
    }
}

// GetFileConfigNames - determine the names of the features and labels sections in the config file
// features - [in,out] a vector of feature name strings
// labels - [in,out] a vector of label name strings
void ConfigHelper::GetDataNamesFromConfig(
    vector<wstring>& features,
    vector<wstring>& labels,
    vector<wstring>& hmms,
    vector<wstring>& lattices)
{
    for (const auto& id : m_config.GetMemberIds())
    {
        if (!m_config.CanBeConfigRecord(id))
            continue;
        const ConfigParameters& temp = m_config(id);
        // see if we have a config parameters that contains a "file" element, it's a sub key, use it
        if (temp.ExistsCurrent(L"scpFile"))
        {
            features.push_back(id);
        }
        else if (temp.ExistsCurrent(L"mlfFile") || temp.ExistsCurrent(L"mlfFileList"))
        {
            labels.push_back(id);
        }
        else if (temp.ExistsCurrent(L"phoneFile"))
        {
            hmms.push_back(id);
        }
        else if (temp.ExistsCurrent(L"denlatTocFile"))
        {
            lattices.push_back(id);
        }
    }
}

ElementType ConfigHelper::GetElementType()
{
    string precision = m_config.Find("precision", "float");
    if (AreEqualIgnoreCase(precision, "float"))
    {
        return ElementType::tfloat;
    }

    if (AreEqualIgnoreCase(precision, "double"))
    {
        return ElementType::tdouble;
    }

    RuntimeError("Not supported precision '%s'. Expected 'double' or 'float'.", precision.c_str());
}

size_t ConfigHelper::GetFeatureDimension()
{
    if (m_config.Exists(L"dim"))
    {
        return m_config(L"dim");
    }

    InvalidArgument("Features must specify dimension: 'dim' property is missing.");
}

size_t ConfigHelper::GetLabelDimension()
{
    if (m_config.Exists(L"labelDim"))
    {
        return m_config(L"labelDim");
    }

    if (m_config.Exists(L"dim"))
    {
        return m_config(L"dim");
    }

    InvalidArgument("Labels must specify dimension: 'dim/labelDim' property is missing.");
}

vector<wstring> ConfigHelper::GetMlfPaths()
{
    vector<wstring> result;
    if (m_config.ExistsCurrent(L"mlfFile"))
    {
        result.push_back(m_config(L"mlfFile"));
    }
    else
    {
        if (!m_config.ExistsCurrent(L"mlfFileList"))
        {
            InvalidArgument("Either mlfFile or mlfFileList must exist in the reader configuration.");
        }

        wstring list = m_config(L"mlfFileList");
        for (msra::files::textreader r(list); r;)
        {
            result.push_back(r.wgetline());
        }
    }

    return result;
}

size_t ConfigHelper::GetRandomizationWindow()
{
    size_t result = randomizeAuto;

    if (m_config.Exists(L"randomize"))
    {
        wstring randomizeString = m_config.CanBeString(L"randomize") ? m_config(L"randomize") : wstring();
        if (!_wcsicmp(randomizeString.c_str(), L"none"))
        {
            result = randomizeNone;
        }
        else if (!_wcsicmp(randomizeString.c_str(), L"auto"))
        {
            result = randomizeAuto;
        }
        else
        {
            result = m_config(L"randomize");
        }
    }

    return result;
}

wstring ConfigHelper::GetRandomizer()
{
    // get the read method, defaults to "blockRandomize"
    wstring randomizer(m_config(L"readMethod", L"blockRandomize"));

    if (randomizer == L"blockRandomize" && GetRandomizationWindow() == randomizeNone)
    {
        InvalidArgument("'randomize' cannot be 'none' when 'readMethod' is 'blockRandomize'.");
    }

    return randomizer;
}

vector<wstring> ConfigHelper::GetSequencePaths()
{
    wstring scriptPath = m_config(L"scpFile");
    wstring rootPath = m_config(L"prefixPathInSCP", L"");

    vector<wstring> filelist;
    fprintf(stderr, "Reading script file %ls ...", scriptPath.c_str());

    // TODO: possibly change to class File, we should be able to read data from pipelines.E.g.
    //  scriptPath = "gzip -c -d FILE.txt |", or do a popen with C++ streams, so that we can have a generic open function that returns an ifstream.
    ifstream scp(msra::strfun::utf8(scriptPath).c_str());
    string line;
    while (getline(scp, line))
    {
        filelist.push_back(msra::strfun::utf16(line));
    }

    fprintf(stderr, " %d entries\n", static_cast<int>(filelist.size()));

    // post processing file list :
    //  - if users specified PrefixPath, add the prefix to each of path in filelist
    //  - else do the dotdotdot expansion if necessary
    if (!rootPath.empty()) // use has specified a path prefix for this  feature
    {
        // first make slash consistent (sorry for Linux users:this is not necessary for you)
        replace(rootPath.begin(), rootPath.end(), L'\\', L'/');

        // second, remove trailing slash if there is any
        wregex trailer(L"/+$");
        rootPath = regex_replace(rootPath, trailer, wstring());

        // third, join the rootPath with each entry in filelist
        if (!rootPath.empty())
        {
            for (wstring& path : filelist)
            {
                if (path.find_first_of(L'=') != wstring::npos)
                {
                    vector<wstring> strarr = msra::strfun::split(path, L"=");
#ifdef WIN32
                    replace(strarr[1].begin(), strarr[1].end(), L'\\', L'/');
#endif
                    path = strarr[0] + L"=" + rootPath + L"/" + strarr[1];
                }
                else
                {
#ifdef WIN32
                    replace(path.begin(), path.end(), L'\\', L'/');
#endif
                    path = rootPath + L"/" + path;
                }
            }
        }
    }
    else
    {
        /*
                do "..." expansion if SCP uses relative path names
                "..." in the SCP means full path is the same as the SCP file
                for example, if scp file is "//aaa/bbb/ccc/ddd.scp"
                and contains entry like
                .../file1.feat
                .../file2.feat
                etc.
                the features will be read from
                //aaa/bbb/ccc/file1.feat
                //aaa/bbb/ccc/file2.feat
                etc.
                This works well if you store the scp file with the features but
                do not want different scp files everytime you move or create new features
                */
        wstring scpDirCached;
        for (auto& entry : filelist)
        {
            ExpandDotDotDot(entry, scriptPath, scpDirCached);
        }
    }

    return filelist;
}

intargvector ConfigHelper::GetNumberOfUtterancesPerMinibatchForAllEppochs()
{
    intargvector numberOfUtterances = m_config(L"nbruttsineachrecurrentiter", ConfigParameters::Array(intargvector(vector<int>{1})));
    for (int i = 0; i < numberOfUtterances.size(); i++)
    {
        if (numberOfUtterances[i] < 1)
        {
            LogicError("nbrUttsInEachRecurrentIter cannot be less than 1.");
        }
    }

    return numberOfUtterances;
}

void ConfigHelper::ExpandDotDotDot(wstring& featPath, const wstring& scpPath, wstring& scpDirCached)
{
    wstring delim = L"/\\";

    if (scpDirCached.empty())
    {
        scpDirCached = scpPath;
        wstring tail;
        auto pos = scpDirCached.find_last_of(delim);
        if (pos != wstring::npos)
        {
            tail = scpDirCached.substr(pos + 1);
            scpDirCached.resize(pos);
        }
        if (tail.empty()) // nothing was split off: no dir given, 'dir' contains the filename
            scpDirCached.swap(tail);
    }
    size_t pos = featPath.find(L"...");
    if (pos != featPath.npos)
        featPath = featPath.substr(0, pos) + scpDirCached + featPath.substr(pos + 3);
}

}}}
