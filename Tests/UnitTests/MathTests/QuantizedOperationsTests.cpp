//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "../../../Source/Math/QuantizedOperations.h"
#include "../../../Source/Math/Helpers.h"

using namespace Microsoft::MSR::CNTK;
namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(QuantizedOperationsUnitTests)

BOOST_FIXTURE_TEST_CASE(MultiplyIntToShort, RandomSeedFixture)
{
    // A[m,k]*B[k,n] = C[m,n]
    int m = 5, n = 4, k = 3;
    std::vector<float> A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}; 
    std::vector<float> B = {16,17,18,19,20,21,22,23,24,25,26,27}; 
    std::vector<float> C_expected = { 316, 367, 418, 469, 520, 370, 430, 490, 550, 610, 424, 493, 562, 631, 700, 478, 556, 634, 712, 790 };
    std::vector<float> C;
    C.resize(m*n);

    shared_ptr<QuantizerBase<float, short>> quantA(new SymmetricQuantizer<float, short>(1));
    shared_ptr<QuantizerBase<float, short>> quantB(new SymmetricQuantizer<float, short>(2));

    // A - is constant; B - is not
    QuantizedMultiplier<float> mult(quantA, true, quantB, false);

    // First pass
    mult.Multiply(m, n, k, A.data(), B.data(), C.data());

    for (size_t i = 0; i < m*n; i++)
        BOOST_CHECK_EQUAL(round(C[i]), C_expected[i]);

    // Second pass, the same matrices
    mult.Multiply(m, n, k, A.data(), B.data(), C.data());

    for (size_t i = 0; i < m*n; i++)
        BOOST_CHECK_EQUAL(round(C[i]), C_expected[i]);

    // Third pass with updated B (size and values)
    int n_upd = 5;
    std::vector<float> B_upd = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    std::vector<float> C_expected_upd = { 46, 52, 58, 64, 70, 100, 115, 130, 145, 160, 154, 178, 202, 226, 250, 208, 241, 274, 307, 340, 262, 304, 346, 388, 430};
    std::vector<float> C_upd;
    C_upd.resize(m*n_upd);
    mult.Multiply(m, n_upd, k, A.data(), B_upd.data(), C_upd.data());
    for (size_t i = 0; i < m*n_upd; i++)
        BOOST_CHECK_EQUAL(round(C_upd[i]), C_expected_upd[i]);
}


BOOST_AUTO_TEST_SUITE_END()

} } } }