//
// <copyright file="Actions.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// This file represents the beginning of moving actions out from CNTK.cpp to make them accessible as a library. To be continued...

#include "ScriptableObjects.h"
#include "File.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

    class ActionsBase : public ScriptableObjects::Object
    {
    };

}}}
