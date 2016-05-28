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
        if (config.ExistsCurrent(L"input"))
        {
            const ConfigParameters& input = config(L"input");

            map<string, wstring> aliasToInputMap;
            for (const pair<string, ConfigParameters>& section : input)
            {
                ConfigParameters input = section.second;
                wstring name = msra::strfun::utf16(section.first);

                // alias is optional
                if (input.ExistsCurrent(L"alias"))
                {
                }
            }
        }

        m_filepath = msra::strfun::utf16(config(L"file"));

        // EvalActions inserts randomize = "none" into the reader config in DoWriteOutoput.
        wstring randomizeString = config(L"randomize", wstring());
        if (!_wcsicmp(randomizeString.c_str(), L"none"))
        {
            m_randomizationWindow = randomizeNone;
        }
        else
        {
            bool randomize = config(L"randomize", true);

            if (!randomize)
            {
                m_randomizationWindow = randomizeNone;
            }
            else if (config.Exists(L"randomizationWindow"))
            {
                m_randomizationWindow = config(L"randomizationWindow");
            }
            else
            {
                m_randomizationWindow = randomizeAuto;
            }
        }

        m_traceLevel = config(L"traceLevel", 1);
    }

}}}
