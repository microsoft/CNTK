//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

namespace CNTK {

    static size_t const g_infinity = SIZE_MAX;

    static size_t const g_dataSweep = SIZE_MAX - 1;

    static size_t const g_32MB = 32 * 1024 * 1024;

    static size_t const g_64MB = g_32MB * 2;

    static size_t const g_4GB = 0x100000000L;

    const static char g_rowDelimiter = '\n';

    const static wchar_t* g_minibatchSourcePosition = L"minibatchSourcePosition";
}
