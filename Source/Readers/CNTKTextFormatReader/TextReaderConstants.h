//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

    const char SPACE_CHAR = ' ';
    const char TAB_CHAR = '\t';

    const char NAME_PREFIX = '|';
    
    const char INDEX_DELIMITER = ':';

    const char ROW_DELIMITER = '\n';
    
    const char ESCAPE_SYMBOL = '#';

    const auto BUFFER_SIZE = 2 * 1024 * 1024;

    inline bool isPrintable(char c)
    {
        return c >= SPACE_CHAR;
    }

    inline bool isNonPrintable(char c)
    {
        return !isPrintable(c);
    }

    inline bool isValueDelimiter(char c)
    {
        return c == SPACE_CHAR || c == TAB_CHAR;
    }

    inline bool isColumnDelimiter(char c)
    {
        return isValueDelimiter(c) || (isNonPrintable(c) && c != ROW_DELIMITER);
    }

}}}
