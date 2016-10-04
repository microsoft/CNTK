//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "TrainingNodes.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template<class ElemType>
void RandomSampleNode<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node                         = dynamic_pointer_cast<RandomSampleNode<ElemType>>(nodeP);
        node->m_allowDuplicates           = m_allowDuplicates;
        node->m_estimateInSampleFrequency = m_estimateInSampleFrequency;
        node->m_sizeOfSampledSet          = m_sizeOfSampledSet;
        node->m_randomSeed                = m_randomSeed;
    }
}

template<class ElemType>
void RandomSampleNode<ElemType>::Save(File& fstream) const
{
    Base::Save(fstream);
    fstream << m_allowDuplicates;
    fstream << m_estimateInSampleFrequency;
    fstream << m_sizeOfSampledSet;
}

template<class ElemType>
void RandomSampleNode<ElemType>::Load(File& fstream, size_t modelVersion)
{
    Base::Load(fstream, modelVersion);
    fstream >> m_allowDuplicates;
    fstream >> m_estimateInSampleFrequency;
    fstream >> m_sizeOfSampledSet;
}

template<class ElemType>
double RandomSampleNode<ElemType>::EstimateInSampleFrequency(double p, double estimatedNumTries) const
 {
     if (m_allowDuplicates)
     {
         return p * m_sizeOfSampledSet;
     }
     else /* No duplicates allowed. Estimated count is same as probability of inclusion. */
     {
         return -expm1(estimatedNumTries * log1p(-p));
     }
 }

template<class ElemType>
void RandomSampleNode<ElemType>::ForwardPropNonLooping()
{
    UpdateWeightsPrefixSum();
    Matrix<ElemType>& valueMatrix = ValueAsMatrix();
    valueMatrix.TransferToDeviceIfNotThere(CPUDEVICE, /*ismoved =*/ true/*means: BOTH state not ok */, /*emptyTransfer =*/ true, /*updatePreferredDevice =*/ false);
    valueMatrix.SetDevice(CPUDEVICE);

    if (m_estimateInSampleFrequency) // W are in the mode the estimating the expected frequency of each class in the sampled set.
    {
        valueMatrix.SwitchToMatrixType(DENSE, matrixFormatDense, false);
        double sumOfWeights = m_samplingWeightsPrefixSum.back();
        const Matrix<ElemType>& samplingWeights = Input(0)->ValueAsMatrix();

        double estimatedNumTries = EstimateNumberOfTries();

        for (int i = 0; i < m_samplingWeightsPrefixSum.size(); i++)
        {
            // Get the sampling probablility for from the weights for i-th class.
            double samplingProb = samplingWeights.GetValue(i, 0) / sumOfWeights;
            double estimatedCount = EstimateInSampleFrequency(samplingProb, estimatedNumTries);
            valueMatrix.SetValue(i, 0, (ElemType)estimatedCount);
        }
    }
    else /* compute random samples */
    {

        valueMatrix.SwitchToMatrixType(SPARSE, matrixFormatSparseCSC, false);
        valueMatrix.Reset();

        // Get vector with indices of randomly sampled classes
        const std::vector<size_t> samples = GetWeightedSamples();

        // Set columns of (sparse) result matrix as indicator vectors
        for (size_t i = 0; i < m_sizeOfSampledSet; i++)
        {
            int sample = samples[i];
            valueMatrix.SetValue(sample, i, 1);
        }
    }
}

template<class ElemType>
void RandomSampleNode<ElemType>::UpdateWeightsPrefixSum()
{
    const Matrix<ElemType>& samplingWeights = Input(0)->ValueAsMatrix();
    m_samplingWeightsPrefixSum.clear();
    double runningWeightsSum = 0;
    for (int iClass = 0; iClass < samplingWeights.GetNumRows(); iClass++)
    {
        ElemType currentWeight = samplingWeights.GetValue(iClass, 0);
        runningWeightsSum += currentWeight;
        m_samplingWeightsPrefixSum.push_back(runningWeightsSum);
    }
}

template<class ElemType>
const std::vector<size_t> RandomSampleNode<ElemType>::GetWeightedSamples()
{
    long dummy;
    // Here we are not interested in the number of sampling tries needed, which is returned in the parameter.
    return RunSampling(dummy);
}

// Estimate the number of tries needed to find sizeOfSampledSet samples
template<class ElemType>
double RandomSampleNode<ElemType>::EstimateNumberOfTries()
{
    // We estimate the average numver of tries by repeating a fixed number of experiments
    const size_t numExperiments = 10; // We choose 10 without any deep justification.
    long totalTries = 0;
    for (int iExperiment = 0; iExperiment < numExperiments; iExperiment++)
    {
        long nTries;
        RunSampling(nTries);
        totalTries += nTries;
    }
    return totalTries / (double)numExperiments;
}

// Runs the sampling returning a vector with the id's of the samples. The parameter nTries is used to return the number of draws that was needed
// to get the expected number of samples.
template<class ElemType>
const std::vector<size_t> RandomSampleNode<ElemType>::RunSampling(long& nTries)
{
    std::uniform_real_distribution<double> r(0, m_samplingWeightsPrefixSum.back());
    std::unordered_set<int> alreadySampled;
    std::vector<size_t> samples;
    CPURNGHandle* cpuRNGHandle = dynamic_cast<CPURNGHandle*>(&GetRNGHandle(CPUDEVICE));
    // find random samples using the specified weight

    if (m_allowDuplicates)
        nTries = m_sizeOfSampledSet;
    else
        nTries = 0; // just initialize and count how many tries we need.

    while (samples.size() < m_sizeOfSampledSet)
    {
        double randomValue = r(cpuRNGHandle->Generator());
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
    return samples;
}

template<class ElemType>
void RandomSampleNode<ElemType>::Validate(bool isFinalValidationPass)
{
    Base::Validate(isFinalValidationPass);
    m_pMBLayout = nullptr;

    let& shape = Input(0)->GetSampleLayout();
    let dims = shape.GetDims();

    size_t nClasses = dims[0];

    if (m_estimateInSampleFrequency)
    {
        // Output one vector containing the estimated in sample frequency for each class.
        SetDims(TensorShape(nClasses, 1), false);
    }
    else /* sampling mode */
    {
        // Output one vector containing the estimated inclusion probability for each class.
        SetDims(TensorShape(nClasses, m_sizeOfSampledSet), false);
    }
}

template<class ElemType>
bool RandomSampleNode<ElemType>::IsOutOfDateWrtInputs() const
{
    // The only input is the sampling weight vector that typically will be constant. 
    if (m_estimateInSampleFrequency)
    {
        // If we are in the mode to estimate the inclusion probabilties for each class we don't need to recompute as long as the input doesn't change.
        return Base::IsOutOfDateWrtInputs();
    }
    else
    {
        // If we are in the mode to generate random samples (i.e. m_estimateInSampleFrequency == false) 
        // we need to recompute the result for each mini-batch even if the weight vector didn't change.
        return true;
    }
}

template class RandomSampleNode<float>;
template class RandomSampleNode<double>;
}}}