//
// <copyright file="Actions.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "ScriptableObjects.h"
#include "File.h"

#include <map>
#include <string>
#include <stdexcept>
#include <list>
#include <vector>

namespace Microsoft { namespace MSR { namespace CNTK {

    class ActionsBase : public ScriptableObjects::Object        //, public ScriptableObjects::CanDo   --we call Do() method on actions
    {
    };

}}}
