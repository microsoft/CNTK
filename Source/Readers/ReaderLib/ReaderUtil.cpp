//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Config.h"
#include "DataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    
    size_t GetRandomizationWindowFromConfig(const ConfigParameters& config)
    {
        wstring randomizeString = config(L"randomize", wstring());
        if (!_wcsicmp(randomizeString.c_str(), L"none")) // TODO: don't support case-insensitive option strings in the new reader
        {
            // "none" is only accepted to be backwards-compatible (DoWriteOutput() in EvalActions.cpp
            // inserts this magic constant into the reader config to prevent it from shuffling the input).
            // In user-defined configurations, 'randomize' should be a boolean.
            return randomizeNone;
        }

        bool randomize = config(L"randomize", true);

        if (!randomize)
        {
            return randomizeNone;
        }

        if (config.Exists(L"randomizationWindow"))
        {
            return config(L"randomizationWindow");
        }

        return randomizeAuto;
    }

}}}
