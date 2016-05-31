//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "CNTKLibrary.h"

namespace CNTK
{
    Value::Value(const NDArrayViewPtr& data)
        : m_data(data)
    {
    }

    /*virtual*/ Value::~Value()
    {
    }

    /*virtual*/ NDArrayViewPtr Value::Data() const
    {
        if (m_data == nullptr)
            LogicError("The data NDArrayView underlying this 'Value' object is null!");

        return m_data;
    }
}
