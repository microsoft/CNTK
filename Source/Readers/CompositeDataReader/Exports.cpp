//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include "Basics.h"

#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "CompositeDataReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new CompositeDataReader("float");
}

extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new CompositeDataReader("double");
}

}}}
