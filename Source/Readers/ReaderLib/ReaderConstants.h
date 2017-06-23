//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

    static size_t const g_infinity = SIZE_MAX;

    static size_t const g_dataSweep = SIZE_MAX - 1;

    static size_t const g_1MB = 1 << 20;

    static size_t const g_2MB = g_1MB << 1;

    static size_t const g_32MB = g_1MB << 5;

    static size_t const g_64MB = g_1MB << 6;

    static size_t const g_4GB = 0x100000000L;

    const static char g_eol = '\n';

}}}
