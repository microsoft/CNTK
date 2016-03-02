//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DataReader.cpp : Defines the exported functions for the DLL application.
//
// TODO: Rename to Exports.cpp
//

#include "stdafx.h"
#include "basetypes.h"

#include "htkfeatio.h" // for reading HTK features
//#include "latticearchive.h"             // for reading HTK phoneme lattices (MMI training)
#include "simplesenonehmm.h" // for MMI scoring
//#include "msra_mgram.h"                 // for unigram scores of ground-truth path in sequence training

#include "rollingwindowsource.h" // minibatch sources
//#include "readaheadsource.h"
#include "chunkevalsource.h"
#define DATAREADER_EXPORTS
#include "DataReader.h"
#include "HTKMLFReader.h"
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

extern "C" DATAREADER_API void GetReaderF(IDataReader** preader)
{
    *preader = new HTKMLFReader<float>();
}
extern "C" DATAREADER_API void GetReaderD(IDataReader** preader)
{
    *preader = new HTKMLFReader<double>();
}

// Utility function, in ConfigFile.cpp, but HTKMLFReader doesn't need that code...

// Trim - trim white space off the start and end of the string
// str - string to trim
// NOTE: if the entire string is empty, then the string will be set to an empty string
/*  void Trim(std::string& str)
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
        str.erase(found+1);
}*/
} } }
