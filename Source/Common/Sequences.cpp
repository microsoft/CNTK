//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#include "Sequences.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Todo: After upgrade to VS2015, remove them after both statics are moved into SetUnqiueAxisName as local static variables.
    std::mutex MBLayout::s_nameIndiciesMutex;
    std::map<std::wstring, size_t> MBLayout::s_nameIndices;
}}}
