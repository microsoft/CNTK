//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"

namespace CNTK
{
    Variable::Variable(const FunctionPtr& function)
        : Variable(function->Output())
    {
			printf("Variable(%s)@%p\n", Name().c_str(), (void*)this);
    }

    FunctionPtr Variable::Owner() const 
    {
        return m_dataFields->m_ownerFunction->shared_from_this(); 
    }
}
