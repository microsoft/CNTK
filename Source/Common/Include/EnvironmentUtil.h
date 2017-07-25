//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

namespace Microsoft { namespace MSR { namespace CNTK {

    class EnvironmentUtil 
    {
    public:
        // Reads and returns an integer value of an environment variable 
        // corresponging to the number of MPI nodes in the current MPI job.
        // This function returns 1 if the variable is not present.
        static int GetTotalNumberOfMPINodes();

        // Reads and returns an integer value of an environment variable 
        // corresponging to the rank of the local MPI node.
        // This function returns 0 if the variable is not present.
        static int GetLocalMPINodeRank();
    };
    
}}}
