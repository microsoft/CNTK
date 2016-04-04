//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <cstdlib>
#include "../../../Source/Math/Matrix.h"

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

template <typename T>
struct Err
{
    static const T Rel;
    static const T Abs;
};
template <>
const float Err<float>::Rel = 1e-5f;
template <>
const double Err<double>::Rel = 1e-5f;
template <>
const float Err<float>::Abs = 1.192092896e-07f;
template <>
const double Err<double>::Abs = 2.2204460492503131e-016;

bool AreEqual(float a, float b, float maxRelError, float maxAbsError);
bool AreEqual(double a, double b, double maxRelError, double maxAbsError);

size_t CountNans(const SingleMatrix& src);

template <typename T>
bool CheckEqual(const Matrix<T>& result, const Matrix<T>& reference, std::string& msg, T maxRelError, T maxAbsError)
{
    std::unique_ptr<T[]> res(result.CopyToArray());
    std::unique_ptr<T[]> ref(reference.CopyToArray());
    int count = 0;
    int badIndex = -1;
    for (int i = 0; i < result.GetNumElements(); ++i)
    {
        if (!AreEqual(res[i], ref[i], maxRelError, maxAbsError) && count++ == 0)
            badIndex = i;
    }
    if (count > 0)
    {
        float a = res[badIndex];
        float b = ref[badIndex];
        std::stringstream ss;
        ss << count << " mismatch" << (count > 1 ? "es" : "") << ", first mismatch at " << badIndex << ", " << a << " != " << b
            << ", rel = " << (std::abs(a - b) / std::max(std::abs(a), std::abs(b))) << ", abs = " << std::abs(a - b);
        msg = ss.str();
    }
    return count == 0;
}

} } } }