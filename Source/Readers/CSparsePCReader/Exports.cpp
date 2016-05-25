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
#include "CSparsePCReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

	extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
	{
		*preader = new CSparsePCReader<float>();
	}
	extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
	{
		*preader = new CSparsePCReader<double>();
	}
}}}
