//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "TrainingNodes.h"
#include <boost/random/uniform_real_distribution.hpp>

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
/*virtual*/ void RandomDistributionNode<ElemType>::ForwardProp(const FrameRange& fr) /*override*/
{
    auto&& result = ValueFor(fr);
    switch (m_type)
    {
    case RandomDistributionType::Uniform:
        result.SetUniformRandomValue(GetRNGHandle(), m_args[0], m_args[1]);
        UpdateRngOffset(GetRngOffset() + result.GetNumElements());
        break;
    case RandomDistributionType::Normal:
        result.SetGaussianRandomValue(GetRNGHandle(), m_args[0], m_args[1]);
        UpdateRngOffset(GetRngOffset() + AsMultipleOf(result.GetNumElements(), 2));
        break;
    case RandomDistributionType::Gumbel:
        result.SetGumbelRandomValue(GetRNGHandle(), m_args[0], m_args[1]);
        UpdateRngOffset(GetRngOffset() + result.GetNumElements());
        break;
    case RandomDistributionType::Bernoulli:
        result.SetUniformRandomMask(ElemType(1 - m_args[0]), ElemType(1), GetRNGHandle());
        UpdateRngOffset(GetRngOffset() + result.GetNumElements());
        break;
    default:
        RuntimeError("RandomDistributionNode::ForwardProp: Unknown random distribution type code %d", m_type);
    }
}


template <class ElemType>
/*virtual*/ void RandomDistributionNode<ElemType>::BackpropTo(const size_t /*inputIndex*/, const FrameRange&) /*override*/
{
    /* Do nothing. The proper fix is for this Node to have it say it does not need gradient. */
}

template <class ElemType>
/*virtual*/ bool RandomDistributionNode<ElemType>::IsOutOfDateWrtInputs() const /*override*/ { return true; }

template class RandomDistributionNode<float>;
template class RandomDistributionNode<double>;
template class RandomDistributionNode<half>;

template<class ElemType>
void RandomSampleNodeBase<ElemType>::Validate(bool isFinalValidationPass)
{
    if (m_sizeOfSampledSet == 0)
    {
        InvalidArgument("Number of requested samples is zero.");
    }

    if (isFinalValidationPass)
    {
        // Sampling without replacement does only work when the number of requested classes is <= number of classes.
        let& shape = Input(0)->GetSampleLayout();
        let dims = shape.GetDims();
        size_t nClasses = dims[0];
        if (!m_allowDuplicates && nClasses <= m_sizeOfSampledSet)
            InvalidArgument("For sampling without duplicates the number of requested samples (%lu) needs to be less than the number of classes (%lu).", m_sizeOfSampledSet, nClasses);
    }
}

template<class ElemType>
void RandomSampleNodeBase<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = dynamic_pointer_cast<RandomSampleNodeBase<ElemType>>(nodeP);
        node->m_allowDuplicates  = m_allowDuplicates;
        node->m_sizeOfSampledSet = m_sizeOfSampledSet;
        node->SetRngState(GetRngSeed(), GetRngOffset());
    }
}

template<class ElemType>
void RandomSampleNodeBase<ElemType>::Save(File& fstream) const
{
    Base::Save(fstream);
    fstream << m_allowDuplicates;
    fstream << m_sizeOfSampledSet;
    RngUser::Save(fstream);
}

template<class ElemType>
void RandomSampleNodeBase<ElemType>::Load(File& fstream, size_t modelVersion)
{
    Base::Load(fstream, modelVersion);
    fstream >> m_allowDuplicates;
    fstream >> m_sizeOfSampledSet;
    RngUser::Load(fstream, modelVersion);
}

template<class ElemType>
void RandomSampleNodeBase<ElemType>::UpdateWeightsPrefixSum()
{
    const Matrix<ElemType>& samplingWeights = Input(0)->ValueAsMatrix();
    m_samplingWeightsPrefixSum.clear();
    double runningWeightsSum = 0;
    for (int iClass = 0; iClass < samplingWeights.GetNumRows(); iClass++)
    {
        ElemType currentWeight = samplingWeights.GetValue(iClass, 0);
        if (currentWeight < 0)
            InvalidArgument("Sampling weights contain negative number %f.", (float)currentWeight);

        runningWeightsSum += (double)currentWeight;
        m_samplingWeightsPrefixSum.push_back(runningWeightsSum);
    }
}

// Runs the sampling returning a vector with the id's of the samples. The parameter nTries is used to return the number of draws that was needed
// to get the expected number of samples.
template<class ElemType>
const std::vector<size_t> RandomSampleNodeBase<ElemType>::RunSampling(size_t& nTries)
{
    boost::random::uniform_real_distribution<double> r(0, m_samplingWeightsPrefixSum.back());
    std::unordered_set<int> alreadySampled;
    std::vector<size_t> samples;
    CPURNGHandle* cpuRNGHandle = dynamic_cast<CPURNGHandle*>(&GetRNGHandle(CPUDEVICE));

    // find random samples using the specified weight
    if (m_allowDuplicates)
        nTries = m_sizeOfSampledSet;
    else
        nTries = 0; // just initialize and count how many tries we need.

    auto offset = GetRngOffset();
    while (samples.size() < m_sizeOfSampledSet)
    {
        double randomValue = r(cpuRNGHandle->Generator());
        offset++;
        // Find the first index where value[idx] >= randomValue.
        auto lower = std::lower_bound(m_samplingWeightsPrefixSum.begin(), m_samplingWeightsPrefixSum.end(), randomValue);
        int idx = (int)(lower - m_samplingWeightsPrefixSum.begin());

        if (m_allowDuplicates)
            samples.push_back(idx);
        else
        {
            // Sampling without replacement: each value can be sampled at most once. 
            // The implementation below using rejection sampling is problematic.
            // E.g if first class has probability p = 0.999 we typically will have to sample 1000 times or more to hit another class.
            // BUGBUG Alternative implementions, e.g:
            // * Weighted Random Sampling with Reservoir: http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf
            // * Binary tree with classes as leafes and branch probs on non-leafes.
            // * As in numpy: https://github.com/numpy/numpy/blob/master/numpy/random/mtrand/mtrand.pyx#L1440
            nTries++;
            if (alreadySampled.find(idx) != alreadySampled.end()) continue;
            else
            {
                samples.push_back(idx);
                alreadySampled.insert(idx);
            }
        }
    }
    UpdateRngOffset(offset);
    return samples;
}

template<class ElemType>
void RandomSampleNode<ElemType>::ForwardPropNonLooping()
{
    Base::UpdateWeightsPrefixSum();

    if (ValueAsMatrix().GetMatrixType() != SPARSE)
    {
        // BUGBUG: matrix type should be configured during validation
        // Note: We allocate a new one instead of switching the type in place since switching in place may
        // affect other nodes who share this matrix due to memory sharing
        auto newSparseValueMatrix = std::make_shared<Matrix<ElemType>>(ValueAsMatrix().GetNumRows(), ValueAsMatrix().GetNumCols(), CPUDEVICE, SPARSE, matrixFormatSparseCSC);
#ifdef _MSC_VER
        ValuePtrRef() = newSparseValueMatrix;
#else
        this->template ValuePtrRef() = newSparseValueMatrix;
#endif
    }

    Matrix<ElemType>& valueMatrix = ValueAsMatrix();

    // TODO: Should we prepare the CSC data directly on the CPU and move it in one go?
    // Currently the reader will place the data onto the GPU. It will then be pulled on-demand to the CPU once (and cached there).
    valueMatrix.TransferToDeviceIfNotThere(CPUDEVICE, /*ismoved =*/ true/*means: BOTH state not ok */, /*emptyTransfer =*/ true, /*updatePreferredDevice =*/ true);
    valueMatrix.Reset();

    // Get vector with indices of randomly sampled classes
    const std::vector<size_t> samples = GetWeightedSamples();

    // Set columns of (sparse) result matrix as indicator vectors
    for (size_t i = 0; i < Base::m_sizeOfSampledSet; i++)
    {
        int sample = samples[i];
        valueMatrix.SetValue(sample, i, 1);
    }
}

template<class ElemType>
const std::vector<size_t> RandomSampleNode<ElemType>::GetWeightedSamples()
{
    size_t dummy;
    // Here we are not interested in the number of sampling tries needed, which is returned in the parameter.
    return Base::RunSampling(dummy);
}

template<class ElemType>
void RandomSampleNode<ElemType>::Validate(bool isFinalValidationPass)
{
    Base::Validate(isFinalValidationPass);
    m_pMBLayout = nullptr;

    let& shape = Input(0)->GetSampleLayout();
    let dims = shape.GetDims();
    size_t numClasses = dims[0];

    // Output: a (sparse) matrix containing m_sizeOfSampledSet columns of 1-hot vectors specifiying the sampled classes.
    SetDims(TensorShape(numClasses, Base::m_sizeOfSampledSet), false);
}

template<class ElemType>
bool RandomSampleNode<ElemType>::IsOutOfDateWrtInputs() const
{
    // We need to recompute the result for each mini-batch even if the weight vector didn't change.
    return true;
}

template class RandomSampleNode<float>;
template class RandomSampleNode<double>;
template class RandomSampleNode<half>;

template<class ElemType>
double RandomSampleInclusionFrequencyNode<ElemType>::EstimateNumberOfTries()
{
    // We estimate the average numver of tries by repeating a fixed number of experiments
    const size_t numExperiments = 10; // We choose 10 without any deep justification.
    long totalTries = 0;
    for (int i = 0; i < numExperiments; i++)
    {
        size_t nTries;
        Base::RunSampling(nTries);
        totalTries += nTries;
    }
    return totalTries / (double)numExperiments;
}

// Estimates the expected number of occurrences of each class in the sampled set.
// For sampling without replacement we use estimate using average number of tries. (Inspired by TensorFlow)
// BUGBUG: Consider to reimplement using a less biased estimate as proposed by Nikos.
template<class ElemType>
double RandomSampleInclusionFrequencyNode<ElemType>::EstimateInSampleFrequency(double p, double estimatedNumTries) const
{
    if (Base::m_allowDuplicates)
    {
        return p * Base::m_sizeOfSampledSet;
    }
    else /* No duplicates allowed. Estimated count is same as probability of inclusion. */
    {
        return -expm1(estimatedNumTries * log1p(-p));
    }
}

template<class ElemType>
void RandomSampleInclusionFrequencyNode<ElemType>::ForwardPropNonLooping()
{
    Base::UpdateWeightsPrefixSum();
    Matrix<ElemType>& valueMatrix = ValueAsMatrix();
    valueMatrix.TransferToDeviceIfNotThere(CPUDEVICE, /*ismoved =*/ true/*means: BOTH state not ok */, /*emptyTransfer =*/ true, /*updatePreferredDevice =*/ false);
    valueMatrix.SetDevice(CPUDEVICE);

    // BUGBUG: matrix type should be configured during validation
    valueMatrix.SwitchToMatrixType(DENSE, matrixFormatDense, false);
    double sumOfWeights = Base::m_samplingWeightsPrefixSum.back();
    const Matrix<ElemType>& samplingWeights = Input(0)->ValueAsMatrix();

    double estimatedNumTries = EstimateNumberOfTries();

    for (int i = 0; i < Base::m_samplingWeightsPrefixSum.size(); i++)
    {
        // Get the sampling probablility for from the weights for i-th class.
        double samplingProb = samplingWeights.GetValue(i, 0) / sumOfWeights;
        double estimatedCount = EstimateInSampleFrequency(samplingProb, estimatedNumTries);
        valueMatrix.SetValue(i, 0, (ElemType)estimatedCount);
    }
}

template<class ElemType>
void RandomSampleInclusionFrequencyNode<ElemType>::Validate(bool isFinalValidationPass)
{
    Base::Validate(isFinalValidationPass);
    m_pMBLayout = nullptr;

    let& shape = Input(0)->GetSampleLayout();
    let dims = shape.GetDims();
    size_t nClasses = dims[0];

    // Output: one vector containing the estimated in sample frequency for each class.
    SetDims(TensorShape(nClasses, 1), false);
}

template class RandomSampleInclusionFrequencyNode<float>;
template class RandomSampleInclusionFrequencyNode<double>;
template class RandomSampleInclusionFrequencyNode<half>;

template<class ElemType>
void DropoutNode<ElemType>::Save(File& fstream) const
{
    Base::Save(fstream);
    RngUser::Save(fstream);
}

template<class ElemType>
void DropoutNode<ElemType>::Load(File& fstream, size_t modelVersion)
{
    Base::Load(fstream, modelVersion);
    RngUser::Load(fstream, modelVersion);
}

#if 0 // outdated version
template<class ElemType>
void BatchNormalizationNode<ElemType>::AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs)
{
    Base::AttachInputs(inputs);

    if (m_pre19SampleCount != 0)
    {
        // copy the sample count loaded from a pre-cntk-19 model into the input parameter.
        Input(RUN_COUNT)->Value().SetValue(ElemType(m_pre19SampleCount));
        // reset the legacy sample count.
        m_pre19SampleCount = 0;
    }
}
#endif

template class DropoutNode<float>;
template class DropoutNode<double>;
template class DropoutNode<half>;

template class BatchNormalizationNode<float>;
template class BatchNormalizationNode<double>;
template class BatchNormalizationNode<half>;

}}}
