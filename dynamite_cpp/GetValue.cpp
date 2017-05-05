//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "GetValue.h"
#include "CNTKLibrary.h"
#include "Variable.h"

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#define let const auto

using namespace std;

namespace CNTK
{
class Memoize
{

public:
    static NDArrayViewPtr GetValue(Variable& v)
    {
        let& fields = *v.m_dataFields;
        fields.m_value;
        return v.Value();
    }
};
}

CNTK::NDArrayViewPtr GetValue(const CNTK::Variable& v)
{
#if 0
    // naive version
    return v.Value();
#else
    return CNTK::Memoize::GetValue(const_cast<CNTK::Variable&>(v));
#endif
}
