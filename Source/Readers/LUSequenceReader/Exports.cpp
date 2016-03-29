//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Exports.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#define DATAWRITER_EXPORTS
#include "LUSequenceReader.h"
#include "LUSequenceWriter.h"

#ifdef _MSC_VER
#include <codecvt>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

void EnableUTF8Support() 
{
#ifdef _MSC_VER
    locale::global(locale(locale::empty(), new codecvt_utf8<wchar_t>));
#else
    locale::global(locale("")); // assuming the user-preferred locale supports UTF-8
#endif
}

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    EnableUTF8Support();
    *preader = new MultiIOBatchLUSequenceReader<float>();
}
extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    EnableUTF8Support();
    *preader = new MultiIOBatchLUSequenceReader<double>();
}

extern "C" DATAWRITER_API void GetWriterF(IDataWriter** pwriter)
{
    *pwriter = new LUSequenceWriter<float>();
}
extern "C" DATAWRITER_API void GetWriterD(IDataWriter** pwriter)
{
    *pwriter = new LUSequenceWriter<double>();
}

}}}
