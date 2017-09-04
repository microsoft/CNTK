//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

class FileWrapper;

namespace CNTK {

// Implementation of a helper class for reading binary files with FileWrapper class
class CBFUtils
{
public:
    static const uint64_t MAGIC_NUMBER = 0x636e746b5f62696eU;

    static void FindMagicOrDie(FileWrapper& f)
    {
        // Read the magic number and make sure we're given a proper CBF file.
        uint64_t number;
        f.ReadOrDie(number);

        if (number != MAGIC_NUMBER)
            RuntimeError("The input (%S) is not a valid CNTK binary format file.",
                f.Filename().c_str());
    }

    static uint32_t GetVersionNumber(FileWrapper& f)
    {
        uint32_t versionNumber;
        f.ReadOrDie(versionNumber);
        return versionNumber;
    }

    static int64_t GetHeaderOffset(FileWrapper& f)
    {
        // Seek to the end of file -8 bytes to find the offset of the header.
        f.SeekOrDie(-int64_t(sizeof(int64_t)), SEEK_END);
        int64_t headerOffset;
        f.ReadOrDie(headerOffset);
        return headerOffset;
    }

private:
    CBFUtils();
};

}
