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

// Implementation of the binary reader.
// Effectively the class represents a factory for connecting the packer,
// transformers and the deserializer together.
class CNTKBinaryFileHelper
{
public:
    static FILE* openOrDie(const string& pathname, const char* mode)
    {
        FILE* f = fopen(pathname.c_str(), mode);
        if (!f)
            RuntimeError("Error opening file '%s': %s.", pathname.c_str(), strerror(errno));
        return f;
    }

    static FILE* openOrDie(const wstring& pathname, const wchar_t* mode)
    {
        FILE* f = _wfopen(pathname.c_str(), mode);
        if (!f)
            RuntimeError("Error opening file '%ls': %s.", pathname.c_str(), strerror(errno));
        return f;
    }

    static void closeOrDie(FILE* f)
    {
        int rc = fclose(f);
        if (rc != 0)
            RuntimeError("Error closing: %s.", strerror(errno));
    }

    static void seekOrDie(FILE* f, int64_t offset, int mode)
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
    
    static int64_t tellOrDie(FILE* f)
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

    static void readOrDie(void* ptr, size_t size, size_t count, FILE* f)
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