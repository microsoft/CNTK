//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full licence information.
//

#include <stdlib.h>
#include <string>
#include "Include/EnvironmentUtil.h"

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// Functions here read environment variables and therefore do not require MPI initialization. 
// Moreover, they can be used without actually loading any MPI libs.

#pragma warning(push)
#pragma warning(disable : 4996) // complains about unsafe getenv.
// However, the way it's used below is safe, since we immediately 
// create an std::string from the returned value.
int EnvironmentUtil::GetTotalNumberOfMPINodes()
{
#if !HAS_MPI
    const char* p = nullptr;
#elif WIN32
    const char* p = getenv("PMI_SIZE");
#else
    const char* p = getenv("OMPI_COMM_WORLD_SIZE");
#endif

    return (!p) ? 1 : stoi(string(p));
}

int EnvironmentUtil::GetLocalMPINodeRank()
{
#if !HAS_MPI
    const char* p = nullptr;
#elif WIN32
    const char* p = getenv("PMI_RANK");
#else
    const char* p = getenv("OMPI_COMM_WORLD_RANK");
#endif

    return (!p) ? 0 : stoi(string(p));
}
#pragma warning(pop)

}}}
