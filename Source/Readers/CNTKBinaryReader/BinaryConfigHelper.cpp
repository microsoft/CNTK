//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <limits>
#include "BinaryConfigHelper.h"
#include "DataReader.h"
#include "StringUtil.h"

using std::string;
using std::wstring;
using std::pair;
using std::vector;
using std::map;

#undef max // max is defined in minwindef.h

namespace Microsoft { namespace MSR { namespace CNTK {

    BinaryConfigHelper::BinaryConfigHelper(const ConfigParameters& config)
    {
        // If the config has an input section, then we need to see if people want to rename streams
        // in the binary file to something else.
        if (config.ExistsCurrent(L"input"))
        {
            const ConfigParameters& input = config(L"input");

            for (const pair<string, ConfigParameters>& section : input)
            {
                ConfigParameters sectionConfig = section.second;
                wstring name = msra::strfun::utf16(section.first);

                // If there is an option for "alias", we will rename the stream with the "alias"
                // name to the target name.
                if (sectionConfig.ExistsCurrent(L"alias"))
                {
                    wstring alias = msra::strfun::utf16(sectionConfig(L"alias"));
                    m_streams[alias] = name;
                }
                else
                    m_streams[name] = name;
            }
        }

        m_filepath = msra::strfun::utf16(config(L"file"));
        m_keepDataInMemory = config(L"keepDataInMemory", false);

        // EvalActions inserts randomize = "none" into the reader config in DoWriteOutoput. We would like this to be true/false,
        // but we can't for this reason. So we will assume false unless we specifically get "true"

        m_randomize = false;
        wstring randomizeString = config(L"randomize", L"false");
        if (!_wcsicmp(randomizeString.c_str(), L"true")) // TODO: don't support case-insensitive option strings in the new reader
            m_randomize = true;

        if (m_randomize)
        {
            if (config.Exists(L"randomizationWindow"))
                m_randomizationWindow = config(L"randomizationWindow");
            else
                m_randomizationWindow = randomizeAuto;
        }
        else
            m_randomizationWindow = randomizeNone;

        m_traceLevel = config(L"traceLevel", 1);
    }

}}}
