//
// <copyright file="Actions.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "Actions.h"
#include "ScriptableObjects.h"
#include "File.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

    // TODO: MakeRuntimeObject will just call procedures, and return a dummy 'bool = true'
    class TrainAction : public ActionsBase
    {
        void Do()
        {
        }
    };

}}}
