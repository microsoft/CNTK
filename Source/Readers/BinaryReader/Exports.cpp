//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Exports.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#define DATAWRITER_EXPORTS
#include "DataWriter.h"
#include "BinaryReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new BinaryReader<float>();
}
extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new BinaryReader<double>();
}

extern "C" DATAWRITER_API void GetWriterF(IDataWriter** pwriter)
{
    *pwriter = new BinaryWriter<float>();
}
extern "C" DATAWRITER_API void GetWriterD(IDataWriter** pwriter)
{
    *pwriter = new BinaryWriter<double>();
}

}}}
