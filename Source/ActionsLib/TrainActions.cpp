//
// <copyright file="Actions.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// This file represents the beginning of moving actions out from CNTK.cpp to make them accessible as a library. To be continued...

#include "Actions.h"
#include "ScriptableObjects.h"
#include "File.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

    class TrainAction : public ActionsBase  // TODO: to be continued...
    {
        void Do()
        {
        }
    };

}}}
