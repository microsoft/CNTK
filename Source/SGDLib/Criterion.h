//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Criterion.h -- helper classes for accumulating criteria

#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "TensorView.h"
#include <memory> // for pair
#include <limits> // for isnan() and numeric_limits  --TODO: is that the right header?

namespace Microsoft { namespace MSR { namespace CNTK {

// helper for criterion pretty printing
static inline string GeneratePaddedFloatOrExpFormat(int padSize, int precision, double value)
{
    char format[16];
    char buffer[512];

    sprintf(format, "%%.%dg", precision);
    sprintf(buffer, format, value);

    for (int i = 0; i < strlen(buffer); i++)
    {
        if (buffer[i] == 'e' || buffer[i] == 'E')
        {
            sprintf(format, "%%%d.%de", padSize, precision);
            return format;
        }
    }
    sprintf(format, "%%%d.%df", padSize, precision);
    return format;
}

// helper class for passing accumulated epoch-level criteria around while retaining their sample counts
// Criteria are represented as a tuple (aggregate criterion, sample count). The average criterion value is their ratio.
struct EpochCriterion : public std::pair<double, size_t>
{
    // construction
    explicit EpochCriterion(double aggregateCriterionValue = 0.0, size_t aggregateSampleCount = 0) : std::pair<double, size_t>(aggregateCriterionValue, aggregateSampleCount) { }
    EpochCriterion(const std::pair<double, size_t>& other) : std::pair<double, size_t>(other) { }

    // main way of reading this out: compute the actual average criterion value from the aggregate and sample count
    double Average() const { return second > 0 ? first / second : 0.0; } // compute the epoch-average

    // a few more handy operations that occured multiple times
    bool IsNan() const { return std::isnan(first); }
    EpochCriterion operator-(const EpochCriterion& other) const { return EpochCriterion(first - other.first, second - other.second); }
    void operator+=(const EpochCriterion& other) { first += other.first; second += other.second; }

    static EpochCriterion Infinity() { return EpochCriterion(std::numeric_limits<double>::infinity()); }
    bool IsInfinity() const { return first == std::numeric_limits<double>::infinity(); }

    // log a criterion value in a form like 'av * count; '
    void LogCriterion(const wstring& name, bool addSemicolon = true) const
    {
        double evalErrorSinceLastLogged = Average();
        int evalSamplesSinceLastLogged  = (int)second;
        fprintf(stderr, "%ls = ", name.c_str());
        string format;
        bool asPercentage = name.back() == 's'; // heuristic: plural forms are error counters
        if (asPercentage)
            fprintf(stderr, (GeneratePaddedFloatOrExpFormat(2, 3, 100*evalErrorSinceLastLogged) + "%%").c_str(), 100*evalErrorSinceLastLogged);
        else
            fprintf(stderr, GeneratePaddedFloatOrExpFormat(0, 8, evalErrorSinceLastLogged).c_str(), evalErrorSinceLastLogged);
        fprintf(stderr, " * %d", evalSamplesSinceLastLogged);
        if (addSemicolon) // if no more numbers follow, then use addSemicolon = false
            fprintf(stderr, "; ");
    }
};

// We accumulate criteria in this struct.
// Criteria are accumulated together with their counts (counts depend on sequence lengths, and different criteria may have different sequence lengths).
template <class ElemType>
struct CriterionAccumulator
{
    // constructor
    CriterionAccumulator(size_t numCriteria, DEVICEID_TYPE deviceId) :
        m_aggregateCriterionValues(make_shared<Matrix<ElemType>> (1, numCriteria, deviceId))
    {
        m_aggregateCriterionValues->SetValue(0);
        m_aggregateSampleCounts.assign(numCriteria, 0);
    }
    // 'i' is the index of the element we add into (multiple eval criteria share the same matrix object)
    // Use 'reset=true' to not accumulate but overwrite.
    const CriterionAccumulator& Add(const std::vector<ComputationNodeBasePtr>& nodes, size_t i, size_t legacyNumSamples)
    {
        return Accumulate</*reset=*/false>(nodes, i, legacyNumSamples);
    }
    const CriterionAccumulator& Assign(const std::vector<ComputationNodeBasePtr>& nodes, size_t i, size_t legacyNumSamples)
    {
        return Accumulate</*reset=*/true>(nodes, i, legacyNumSamples);
    }
    // retrieve an accumulated result as a pair (numerator, denominator)
    EpochCriterion GetCriterion(size_t i) const
    {
        // BUGBUG: For unknown reasons, this (or the other below) check makes a difference for MPI configs.
        //         If it is left out, then training and test configs end up being scaled by the same factor close to 1.
        if (m_aggregateSampleCounts[i] == 0)
            return EpochCriterion(0, 0); // avoid unnecessary GPU access
        else
            return EpochCriterion(m_aggregateCriterionValues->GetValue(0, i), m_aggregateSampleCounts[i]);
    }

private:
    // shared part of Add() and Assign()
    // This code assumes that if number of samples is 0, the criterion value is also 0 and does not need to be fetched from the GPU.
    template<bool reset>
    const CriterionAccumulator& Accumulate(const std::vector<ComputationNodeBasePtr>& nodes, size_t i, size_t legacyNumSamples)
    {
        const auto& node = nodes[i]; // multiple nodes are managed by this struct
        size_t beta = reset ? 0 : 1;
        size_t numSamples = GetNumSamples(nodes[i], legacyNumSamples);

        // For criterion nodes that emit criteria per frame, we will at this point
        // do masking and an implicit reduction.

        // get a TensorView of the criterion values to aggregate
        // TODO: Verify that node->GetSampleLayout().GetNumElements() == 1. Require explicit summation to declare intent that this is a criterion.
        FrameRange fr(node->GetMBLayout());
        node->MaskMissingValueColumnsToZero(fr); // set gaps to zero, so that we can aggregate
        // get a TensorView of our aggregator
        TensorShape shape{ m_aggregateCriterionValues->GetNumRows(), m_aggregateCriterionValues->GetNumCols() };
        shape.NarrowTo(1, i, i + 1); // narrow to the single element that corresponds to the accumulator value
        auto criterionAccumulator = TensorView<ElemType>(m_aggregateCriterionValues, shape);

        if (numSamples > 0) // (if MB is empty, matrix may not have the correct row dmension)
        {
            auto criterionValue = node->As<ComputationNode<ElemType>>()->ValueTensorFor(SIZE_MAX, fr);
            // accumulate
            // Note: If criterion is > [1 x 1] then inverse broadcasting will kick in and aggregate.
            // If count is zero, we lazily consider the numerator as zero as well.
            criterionAccumulator.DoCopyOf(m_aggregateSampleCounts[i] ? (float)beta : 0, criterionValue, 1);
        }
        m_aggregateSampleCounts[i] = m_aggregateSampleCounts[i] * beta + numSamples;
        return *this;
    }

public:
    // get the number of samples
    static size_t GetNumSamples(const ComputationNodeBasePtr& node, size_t legacyNumSamples)
    {
        if (node->HasMBLayout())
            return node->GetMBLayout()->GetActualNumSamples();
        else
            return legacyNumSamples;
    }

private:
    shared_ptr<Matrix<ElemType>> m_aggregateCriterionValues; // [1 x N]
    vector<size_t> m_aggregateSampleCounts;                  // [N]
};

}}}
