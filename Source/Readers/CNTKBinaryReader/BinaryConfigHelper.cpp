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
#include "ReaderConstants.h"
#include "ReaderUtil.h"

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

        string precision = config.Find("precision", "float");
        if (AreEqualIgnoreCase(precision, "double"))
        {
            m_elementType = ElementType::tdouble;
        }
        else if (AreEqualIgnoreCase(precision, "float"))
        {
            m_elementType = ElementType::tfloat;
        }
        else
        {
            RuntimeError("Not supported precision '%s'. Expected 'double' or 'float'.", precision.c_str());
        }

        m_filepath = msra::strfun::utf16(config(L"file"));
        m_keepDataInMemory = config(L"keepDataInMemory", false);

        m_randomizationWindow = GetRandomizationWindowFromConfig(config);
        m_sampleBasedRandomizationWindow = config(L"sampleBasedRandomizationWindow", false);
        if (!m_sampleBasedRandomizationWindow && m_randomizationWindow == randomizeAuto)
        {
            // The size of the chunk for the binary reader is specified in terms of the number of sequences
            // per chunk and is fixed at the time when the data is serialized into the binary format.
            // As a result, the on-disk size of a chunk can be arbitrary, and 32MB number used here is 
            // merely a heuristic. 
            m_randomizationWindow = g_4GB / g_32MB; // 128 chunks. 
        }

        m_traceLevel = config(L"traceLevel", 1);
    }

}}}
