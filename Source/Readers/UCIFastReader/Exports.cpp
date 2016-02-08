//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Exports.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "UCIFastReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void DATAREADER_API GetReader(IDataReader<ElemType>** preader)
{
    *preader = new UCIFastReader<ElemType>();
}

extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader)
{
    GetReader(preader);
}
extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader)
{
    GetReader(preader);
}
} } }
