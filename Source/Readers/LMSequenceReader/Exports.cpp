//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Exports.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#define DATAWRITER_EXPORTS
#include "SequenceReader.h"
#include "SequenceWriter.h"

namespace Microsoft { namespace MSR { namespace CNTK {

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new BatchSequenceReader<float>();
}
extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new BatchSequenceReader<double>();
}

extern "C" DATAWRITER_API void GetWriterF(IDataWriter** pwriter)
{
    *pwriter = new LMSequenceWriter<float>();
}
extern "C" DATAWRITER_API void GetWriterD(IDataWriter** pwriter)
{
    *pwriter = new LMSequenceWriter<double>();
}

}}}
