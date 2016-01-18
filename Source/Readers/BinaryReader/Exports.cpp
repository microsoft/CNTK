//
// <copyright file="Exports.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
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

template <class ElemType>
void DATAREADER_API GetReader(IDataReader<ElemType>** preader)
{
    *preader = new BinaryReader<ElemType>();
}

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    GetReader(preader);
}
extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    GetReader(preader);
}

template <class ElemType>
void DATAWRITER_API GetWriter(IDataWriter<ElemType>** pwriter)
{
    *pwriter = new BinaryWriter<ElemType>();
}

extern "C" DATAWRITER_API void GetWriterF(IDataWriter<float>** pwriter)
{
    GetWriter(pwriter);
}
extern "C" DATAWRITER_API void GetWriterD(IDataWriter<double>** pwriter)
{
    GetWriter(pwriter);
}
}
}
}