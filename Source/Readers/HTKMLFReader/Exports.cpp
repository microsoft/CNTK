//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"

#define DATAREADER_EXPORTS
#define DATAWRITER_EXPORTS
#include "HTKMLFReader.h"
#include "HTKMLFWriter.h"

namespace Microsoft { namespace MSR { namespace CNTK {

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new HTKMLFReader<float>();
}
extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new HTKMLFReader<double>();
}

extern "C" DATAWRITER_API void GetWriterF(IDataWriter** pwriter)
{
    *pwriter = new HTKMLFWriter<float>();
}
extern "C" DATAWRITER_API void GetWriterD(IDataWriter** pwriter)
{
    *pwriter = new HTKMLFWriter<double>();
}

#ifdef _WIN32
// Utility function, in ConfigFile.cpp, but HTKMLFReader doesn't need that code...

// Trim - trim white space off the start and end of the string
// str - string to trim
// NOTE: if the entire string is empty, then the string will be set to an empty string
void Trim(std::string& str)
{
    auto found = str.find_first_not_of(" \t");
    if (found == npos)
    {
        str.erase(0);
        return;
    }
    str.erase(0, found);
    found = str.find_last_not_of(" \t");
    if (found != npos)
        str.erase(found + 1);
}
#endif

}}}
