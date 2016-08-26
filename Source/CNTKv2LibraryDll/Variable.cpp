//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"

namespace CNTK
{
    /*static*/ const std::vector<Axis> Variable::DefaultInputVariableDynamicAxes = { Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() };

    Variable::Variable(const FunctionPtr& function)
        : Variable(function->Output())
    {
    }

    FunctionPtr Variable::Owner() const 
    {
        return m_dataFields->m_ownerFunction->shared_from_this(); 
    }
}
