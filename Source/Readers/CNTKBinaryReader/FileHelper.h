//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#ifndef _CNTKBINARYFILEHELPER_
#define _CNTKBINARYFILEHELPER_

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings
#endif

#define _CRT_NONSTDC_NO_DEPRECATE // make VS accept POSIX functions without _
#pragma warning(disable : 4996)   // ^^ this does not seem to work--TODO: make it work
#define _FILE_OFFSET_BITS 64      // to force fseeko() and ftello() 64 bit in Linux

#include <stdio.h>
#ifdef __WINDOWS__
#endif
#ifdef __unix__
#include <sys/types.h>
#include <sys/stat.h>
#endif
#include <errno.h>
#include <stdint.h>
#include <assert.h>
#include "Basics.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Implementation of a helper class for reading/writing to binary files
// on Windows and Linux
class CNTKBinaryFileHelper
{
public:
    static const uint64_t MAGIC_NUMBER = 0x636e746b5f62696eU;

    static void FindMagicOrDie(FILE* f, wstring name) {
        // Read the magic number and make sure we're given a proper CBF file.
        uint64_t number;
        ReadOrDie(&number, sizeof(number), 1, f);
        if (number != MAGIC_NUMBER)
            RuntimeError("The input (%S) is not a valid CNTK binary format file.",
                name.c_str());
    }

    static uint32_t GetVersionNumber(FILE* f) {
        uint32_t versionNumber;
        ReadOrDie(&versionNumber, sizeof(versionNumber), 1, f);
        return versionNumber;
    }

    static int64_t GetHeaderOffset(FILE* f) {
        // Seek to the end of file -8 bytes to find the offset of the header.
        SeekOrDie(f, -int64_t(sizeof(int64_t)), SEEK_END);
        int64_t headerOffset;
        ReadOrDie(&headerOffset, sizeof(headerOffset), 1, f);
        return headerOffset;
    }

    static FILE* openOrDie(const string& pathname, const char* mode)
    {
        FILE* f = fopen(pathname.c_str(), mode);
        if (!f)
            RuntimeError("Error opening file '%s': %s.", pathname.c_str(), strerror(errno));
        return f;
    }

    static FILE* OpenOrDie(const wstring& pathname, const wchar_t* mode)
    {
        FILE* f = _wfopen(pathname.c_str(), mode);
        if (!f)
            RuntimeError("Error opening file '%ls': %s.", pathname.c_str(), strerror(errno));
        return f;
    }

    static void CloseOrDie(FILE* f)
    {
        int rc = fclose(f);
        if (rc != 0)
            RuntimeError("Error closing: %s.", strerror(errno));
    }

    static void SeekOrDie(FILE* f, int64_t offset, int mode)
    {
        int rc;
#ifdef __WINDOWS__
        rc = _fseeki64(f, offset, mode);
#else
        rc = fseeko(f,offset,mode);
#endif
        if (rc != 0)
            RuntimeError("Error seeking: %s.", strerror(errno));
    }
    
    static int64_t TellOrDie(FILE* f)
    {
        size_t rc;
#ifdef __WINDOWS__
        rc = _ftelli64(f);
#else
        rc = ftello(f);
#endif
        if (rc == -1L)
            RuntimeError("Error telling: %s.", strerror(errno));
        return rc;
    }

    static void ReadOrDie(void* ptr, size_t size, size_t count, FILE* f)
    {
        size_t rc;
        rc = fread(ptr, size, count, f);
        if (rc != count)
            RuntimeError("Error reading: %s.", strerror(errno));
    }

private:
    CNTKBinaryFileHelper();
};

}}}
#endif