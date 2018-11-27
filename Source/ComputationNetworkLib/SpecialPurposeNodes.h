//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "gammacalculation.h"
#include "NonlinearityNodes.h"
#include "latticearchive.h"
#include "ProgressTracing.h"

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <locale>
#include <codecvt>

namespace Microsoft { namespace MSR { namespace CNTK {

// This header collects special-purpose nodes.

// -----------------------------------------------------------------------
// TraceNode (input, say='', enabled=true, gradient=false, showFrequency=10, showFirst=10, format=[]) -- trace a node's value
// Traces a node's value using WriteMinibatchWithFormatting().
// -----------------------------------------------------------------------

template <class ElemType>
class TraceNode : public ComputationNode<ElemType>, public NumInputs<1>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"Trace"; }

public:
    TraceNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    TraceNode(const ScriptableObjects::IConfigRecordPtr configp);
    virtual void Save(File& fstream) const override;
    virtual void Load(File& fstream, size_t modelVersion) override;
    virtual void /*IComputationNode::*/ BeginForwardProp() override;
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override;
    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override;
    virtual void /*ComputationNode::*/ Validate(bool isFinalValidationPass) override;

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

private:
    void Log(const FrameRange& fr, bool logGradientInstead) const;

private:
    // configuration
    std::wstring m_message;
    size_t m_logFrequency = 0; // Note: This can be changed in the debugger on the fly.
    size_t m_logFirst = 0;
    bool m_logGradientToo = false;
    WriteFormattingOptions m_formattingOptions;
    size_t m_onlyUpToRow = SIZE_MAX;
    size_t m_onlyUpToT = SIZE_MAX;
    // cached stuff (not persisted)
    size_t m_numMBsRun = 0;
    std::vector<std::string> m_labelMapping;
};

#ifdef COMING_SOON

// -----------------------------------------------------------------------
// GMMLogLikelihoodNode (unnormedPrior, means, logStdDevs, features) -- GMM log LL over input vector(s)
// calculates the log likelihood of a feature given parameters of a Gaussian mixture model (GMM) with shared diagonal variance
//  - unnormedPrior: mix weights, #rows = #mixture components
//  - means: means, all mix means concatenated  (i.e. dim = feature dim x prior dim)
//  - logStdDevs: std deviations, pooled across mix (i.e. same dim as features)
// UnnormedPrior, means, and logStdDevs can be either a single column or one per sample, e.g.
// when parameters are computed by other nodes.
// -----------------------------------------------------------------------

template <class ElemType>
class GMMLogLikelihoodNode : public ComputationNode<ElemType>, public NumInputs<4>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"GMMLogLikelihood"; }

public:
    DeclareConstructorFromConfigWithNumInputs(GMMLogLikelihoodNode);
    GMMLogLikelihoodNode(DEVICEID_TYPE deviceId, const wstring& name)
        : ComputationNode<ElemType>(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        // get the right slice
        const size_t colsPrior = Input(0)->GetSampleMatrixNumCols();

        Matrix<ElemType> sliceGradientValue = DataFor(*m_gradient, fr);
        Matrix<ElemType> slicePosterior = DataFor(*m_posterior, fr);

        switch (inputIndex)
        {
        case 0:
        {
            if (colsPrior == 1)
                BackpropToUnnormedPrior(Input(0)->Gradient(), sliceGradientValue, *m_prior, slicePosterior, *m_temp);
            else
            {
                Matrix<ElemType> sliceUnnormedPriorGradient = Input(0)->GradientFor(fr);
                Matrix<ElemType> slicePrior = DataFor(*m_prior, fr); // TODO: use the right MBLayout, then we won't need the special case
                BackpropToUnnormedPrior(sliceUnnormedPriorGradient, sliceGradientValue, slicePrior, slicePosterior, *m_temp);
            }
        }
        break;
        case 1:
        {
            Matrix<ElemType> sliceNormedDeviationVectors = DataFor(*m_normedDeviationVectors, fr);
            if (colsPrior == 1)
                BackpropToMean(Input(1)->Gradient(), sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
            else
            {
                Matrix<ElemType> sliceMeanGradient = Input(1)->GradientFor(fr);
                BackpropToMean(sliceMeanGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
            }
        }
        break;
        case 2:
        {
            Matrix<ElemType> sliceNormedDeviation = DataFor(*m_normedDeviation, fr);
            if (colsPrior == 1)
                BackpropToLogStddev(Input(2)->Gradient(), sliceGradientValue, sliceNormedDeviation, slicePosterior, *m_temp);
            else
            {
                Matrix<ElemType> sliceLotStddevGradient = Input(2)->GradientFor(fr);
                BackpropToLogStddev(sliceLotStddevGradient, sliceGradientValue, sliceNormedDeviation, slicePosterior, *m_temp);
            }
        }
        break;
        case 3:
        {
            Matrix<ElemType> sliceNormedDeviationVectors = DataFor(*m_normedDeviationVectors, fr);
            Matrix<ElemType> sliceFeatureGradient = Input(3)->GradientFor(fr);
            BackpropToFeature(sliceFeatureGradient, sliceGradientValue, sliceNormedDeviationVectors, slicePosterior, *m_temp);
        }
        break;
        default:
            InvalidArgument("GMMLogLikelihoodNode criterion only takes four inputs.");
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    void BackpropToUnnormedPrior(Matrix<ElemType>& unnormedPriorGradientValues, const Matrix<ElemType>& gradientValues,
                                 const Matrix<ElemType>& prior, const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        temp.AssignDifferenceOf(posterior, prior);
        temp.RowElementMultiplyWith(gradientValues);
        if (prior.GetNumCols() == posterior.GetNumCols())
            unnormedPriorGradientValues += temp;
        else if (prior.GetNumCols() == 1)
            Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(posterior.GetNumCols(), 1, unnormedPriorGradientValues.GetDeviceId()), false, unnormedPriorGradientValues);
        else
            RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
    }

    void BackpropToMean(Matrix<ElemType>& meanGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
                        Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        size_t numComponent = posterior.GetNumRows();
        size_t numSamples = posterior.GetNumCols();
        size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

        temp.SetValue(normedDeviationVectors); // recall normedDeviationVectors <-- (x-u_c)/(stddev^2)
        temp.Reshape(featureSize, numSamples * numComponent);

        posterior.Reshape(1, numSamples * numComponent);
        temp.RowElementMultiplyWith(posterior); // temp <-- posterior * (x-u_c)/(stddev^2)

        posterior.Reshape(numComponent, numSamples);          // reshape back
        temp.Reshape(featureSize * numComponent, numSamples); // reshape back

        temp.RowElementMultiplyWith(gradientValues);

        if (numSamples == meanGradientValues.GetNumCols())
            meanGradientValues += temp;
        else if (meanGradientValues.GetNumCols() == 1)
            Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, meanGradientValues.GetDeviceId()), false, meanGradientValues);
        else
            RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
    }

    void BackpropToLogStddev(Matrix<ElemType>& logStddevGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviation,
                             const Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        size_t numComponent = posterior.GetNumRows();
        size_t numSamples = posterior.GetNumCols();

        temp.AssignDifferenceOf(normedDeviation, (ElemType) numComponent);
        temp.ElementMultiplyWith(posterior);
        temp.RowElementMultiplyWith(gradientValues);
        if (logStddevGradientValues.GetNumCols() == numSamples)
            logStddevGradientValues += temp;
        else if (logStddevGradientValues.GetNumCols() == 1)
            Matrix<ElemType>::MultiplyAndAdd(temp, false, ConstOnes(numSamples, 1, logStddevGradientValues.GetDeviceId()), false, logStddevGradientValues);
        else
            RuntimeError("GMMLogLikelihoodNode: stddev should either have same number of columns as the features or have only one column.");
    }

    void BackpropToFeature(Matrix<ElemType>& featureGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& normedDeviationVectors,
                           Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        size_t numComponent = posterior.GetNumRows();
        size_t numSamples = posterior.GetNumCols();
        size_t featureSize = normedDeviationVectors.GetNumRows() / numComponent;

        temp.SetValue(normedDeviationVectors);
        temp *= -1;
        temp.Reshape(featureSize, numSamples * numComponent);
        posterior.Reshape(1, numSamples * numComponent);
        temp.RowElementMultiplyWith(posterior);

        posterior.Reshape(numComponent, numSamples);
        temp.Reshape(featureSize * numComponent, numSamples);
        temp.RowElementMultiplyWith(gradientValues);

        for (int i = 0; i < numComponent; i++)
            featureGradientValues.AddWithRowSliceValuesOf(temp, i * featureSize, featureSize);
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();

        size_t numCols = Input(3)->GetSampleMatrixNumCols();
        size_t numComponents = Input(0)->GetSampleMatrixNumRows();
        size_t colsPrior = Input(0)->GetSampleMatrixNumCols(); // may be 1
        size_t featureSize = Input(3)->GetSampleMatrixNumRows();

        m_prior->Resize(numComponents, colsPrior);
        m_stddev->Resize(numComponents, colsPrior);
        m_normedDeviation->Resize(numComponents, numCols);
        m_normedDeviationVectors->Resize(numComponents * featureSize, numCols);
        m_posterior->Resize(numComponents, numCols);
    }

    // input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        size_t colsPrior = Input(0)->GetSampleMatrixNumCols();
        size_t numSamples = Input(3)->GetSampleMatrixNumCols();

        // get the right slice
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);
        Matrix<ElemType> sliceFeature = Input(3)->ValueFor(fr);
        Matrix<ElemType> sliceNormedDeviation = DataFor(*m_normedDeviation, fr);
        Matrix<ElemType> sliceNormedDeviationVectors = DataFor(*m_normedDeviationVectors, fr);
        Matrix<ElemType> slicePosterior = DataFor(*m_posterior, fr);

        if (colsPrior == 1)
        {
            ForwardPropS(sliceOutputValue, Input(0)->Value(), Input(1)->Value(), Input(2)->Value(), sliceFeature,
                         *m_prior, *m_stddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, *m_temp);
        }
        else if (colsPrior == numSamples)
        {
            Matrix<ElemType> sliceUnnormedPrior = Input(0)->ValueFor(fr);
            Matrix<ElemType> sliceMean = Input(1)->ValueFor(fr);
            Matrix<ElemType> sliceLogstddev = Input(2)->ValueFor(fr);

            Matrix<ElemType> slicePrior = DataFor(*m_prior, fr);
            Matrix<ElemType> sliceStddev = DataFor(*m_stddev, fr);

            ForwardPropS(sliceOutputValue, sliceUnnormedPrior, sliceMean, sliceLogstddev, sliceFeature,
                         slicePrior, sliceStddev, sliceNormedDeviationVectors, sliceNormedDeviation, slicePosterior, *m_temp);
        }
        else // should not reach the code since validation should fail already
            RuntimeError("GMMLogLikelihoodNode: UnnormedPrior should either have same number of columns as the features or have only one column.");
    }

    // input0=unnormedPrior, input1=mean, input2=logstddev, input3=feature
    // If we want to speed up we need to replace following code with a several specialized GPU functions
    /*TODO: merge with call site*/ void ForwardPropS(Matrix<ElemType>& functionValues, const Matrix<ElemType>& unnormedPrior, const Matrix<ElemType>& mean, Matrix<ElemType>& logstddev,
                                                     const Matrix<ElemType>& feature, Matrix<ElemType>& prior, Matrix<ElemType>& stddev, Matrix<ElemType>& normedDeviationVectors,
                                                     Matrix<ElemType>& normedDeviation, Matrix<ElemType>& posterior, Matrix<ElemType>& temp)
    {
        int numComponent = unnormedPrior.GetNumRows();
        size_t numSamples = feature.GetNumCols();
        size_t featureDim = feature.GetNumRows();

        // compute prior which is softmax of unnormedPrior
        prior.AssignLogSoftmaxOf(unnormedPrior, true); // log prior

        prior.InplaceExp();

        // compute stddev
        stddev.AssignExpOf(logstddev);

#if DUMPOUTPUT
        unnormedPrior.Print("unnormedPrior", 0, min(5, unnormedPrior.GetNumRows() - 1), 0, min(10, unnormedPrior.GetNumCols() - 1));
        mean.Print("mean", 0, min(5, mean.GetNumRows() - 1), 0, min(10, mean.GetNumCols() - 1));
        logstddev.Print("logstddev", 0, min(5, logstddev.GetNumRows() - 1), 0, min(10, logstddev.GetNumCols() - 1));

        prior.Print("prior", 0, min(5, prior.GetNumRows() - 1), 0, min(10, prior.GetNumCols() - 1));
        stddev.Print("stddev", 0, min(5, stddev.GetNumRows() - 1), 0, min(10, stddev.GetNumCols() - 1));
#endif

        // compute normedDeviation <-- ||x-u_c||^2/(stddev^2)
        normedDeviationVectors.AssignRepeatOf(feature, numComponent, 1);
        normedDeviationVectors -= mean;                                        // each column of the mean has multiple mean components
        normedDeviationVectors.Reshape(featureDim, numSamples * numComponent); // now each column is feature-mean_i

        normedDeviation.AssignVectorNorm2Of(normedDeviationVectors, true);
        normedDeviation ^= 2;
        temp.AssignRepeatOf(stddev, 1, numSamples / stddev.GetNumCols()); // stddev.GetNumCols() is either 1 or =numSamples
        temp.Reshape(1, temp.GetNumElements());                           // one stddev value for each component for each sample
        temp ^= 2;
        normedDeviation.ElementDivideBy(temp); // normedDeviation and temp have same dim (1, numSamples* numComponent)

        // compute  normedDeviationVectors <-- (x-u_c)/(stddev^2)
        normedDeviationVectors.RowElementDivideBy(temp);                       // divide twice
        normedDeviationVectors.Reshape(featureDim * numComponent, numSamples); // reshape back

        // compute per-component likelihood
        posterior.AssignProductOf(-0.5f, normedDeviation); // posterior  <-- -||x-u_c||^2/(stddev^2)/2 and in (1, numSamples* numComponent) dim
        temp.InplaceLog();
        temp *= ((ElemType) numComponent / 2.0f);                   // temp <-- stddev^c and in (1, numSamples* numComponent) dim
        posterior -= temp;                                          // posterior  <-- exp[-||x-u_c||^2/(stddev^2)/2]/(stddev^c)
        posterior -= (ElemType)(numComponent / 2.0f * log(TWO_PI)); // likelihood for each component and sample is now computed and stored in posterior
        posterior.InplaceExp();                                     // posterior  <-- exp(-||x-u_c||^2/(stddev^2)/2)

        normedDeviation.Reshape(numComponent, numSamples); // reshape back
        posterior.Reshape(numComponent, numSamples);       // reshape back

        // compute posterior <-- prior_i * likelihood_i
        if (unnormedPrior.GetNumCols() == numSamples) // each sample has different prior
            posterior.ElementMultiplyWith(prior);
        else // all samples share the same prior
            posterior.ColumnElementMultiplyWith(prior);

        // compute GMM log-likelihood
        Matrix<ElemType>::Multiply(ConstOnes(1, numComponent, posterior.GetDeviceId()), false, posterior, false, functionValues); // functionValues <-- total likelihood
        posterior.RowElementDivideBy(functionValues);                                                                             // posterior <-- per-comp likelihood / total likelihood
        functionValues.InplaceLog();                                                                                              // log likelihood

#if DUMPOUTPUT
        temp.Print("temp", 0, min(5, temp.GetNumRows() - 1), 0, min(10, temp.GetNumCols() - 1));
        normedDeviation.Print("normedDeviation", 0, min(5, normedDeviation.GetNumRows() - 1), 0, min(10, normedDeviation.GetNumCols() - 1));

        posterior.Print("posterior", 0, min(5, posterior.GetNumRows() - 1), 0, min(10, posterior.GetNumCols() - 1));
        functionValues.Print("functionValues", 0, min(5, functionValues.GetNumRows() - 1), 0, min(10, functionValues.GetNumCols() - 1));

        functionValues.Print("GMMLogLikelihoodNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        size_t rows[4];
        for (int i = 0; i < 4; i++)
            rows[i] = Input(i)->GetSampleMatrixNumRows();

        if (isFinalValidationPass)
        {
            if (!Input(3)->HasMBLayout())
                InvalidArgument("GMMLogLikelihoodNode: Features must be a minibatch.");
            if (Input(0)->GetMBLayout() != Input(1)->GetMBLayout() || Input(0)->GetMBLayout() != Input(2)->GetMBLayout())
                InvalidArgument("GMMLogLikelihoodNode: First three arguments must have the same MBLayout (which may be none).");

            if (rows[0] != rows[2])
                LogicError("GMMLogLikelihoodNode: UnnormedPrior (first input) should have same dimension as logStddev (third input), i.e., all dimensions in each Gaussian component share the same stddev.");

            if (rows[1] != rows[0] * rows[3])
                LogicError("GMMLogLikelihoodNode: the number of rows in mean (second input) should equal rows(unnormedPrior(first input) * rows(feature(fourth input)).");
        }

        SetDims(TensorShape(1), true);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<GMMLogLikelihoodNode<ElemType>>(nodeP);
            *node->m_prior = *m_prior;
            *node->m_normedDeviation = *m_normedDeviation;
            *node->m_normedDeviationVectors = *m_normedDeviationVectors;
            *node->m_stddev = *m_stddev;
            *node->m_posterior = *m_posterior;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_prior, matrixPool);
        RequestMatrixFromPool(m_normedDeviation, matrixPool);
        RequestMatrixFromPool(m_normedDeviationVectors, matrixPool);
        RequestMatrixFromPool(m_stddev, matrixPool);
        RequestMatrixFromPool(m_posterior, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_prior, matrixPool);
        ReleaseMatrixToPool(m_normedDeviation, matrixPool);
        ReleaseMatrixToPool(m_normedDeviationVectors, matrixPool);
        ReleaseMatrixToPool(m_stddev, matrixPool);
        ReleaseMatrixToPool(m_posterior, matrixPool);
        ReleaseMatrixToPool(m_temp, matrixPool);
    }

protected:
    shared_ptr<Matrix<ElemType>> m_prior;
    shared_ptr<Matrix<ElemType>> m_normedDeviation;
    shared_ptr<Matrix<ElemType>> m_normedDeviationVectors;
    shared_ptr<Matrix<ElemType>> m_stddev;
    shared_ptr<Matrix<ElemType>> m_posterior;
    shared_ptr<Matrix<ElemType>> m_temp;
};

template class GMMLogLikelihoodNode<float>;
template class GMMLogLikelihoodNode<double>;

#endif

// -----------------------------------------------------------------------
// SequenceWithSoftmaxNode (label, prediction, loglikelihood)
// word-lattice based sequence training criterion, using a Microsoft-proprietary lattice format
//
// This node is likely not very useful for external use since it uses an MS-proprietary lattice-archive format
// that requires Frank's DBN.exe tool to create. The inner C++ code for converting HTK lattices
// into this format is in this repo (latticearchive.h), but not the outer main program.
// -----------------------------------------------------------------------

template <class ElemType>
class SequenceWithSoftmaxNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SequenceWithSoftmax";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(SequenceWithSoftmaxNode);
    SequenceWithSoftmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_gammaCalcInitialized(false), m_invalidMinibatch(false)
    {
    }

    // compute gradients to input observations, the weights to the observations, and the class log posterior probabilities
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        // auto t_start_time = Timer::MilliSecondElapsed();
        // left Node must be a scalar
        if (inputIndex == 0) // left derivative
        {
            BackpropToLeft(*m_logSoftmaxOfRight, Input(inputIndex)->Gradient(), Gradient());
        }
        else if (inputIndex == 1)
        {
            if (m_invalidMinibatch)
            {
                Input(inputIndex)->Gradient().SetValue(0.0f);
                Value().SetValue(1.0f);
            }
            else
            {
                FrameRange fr(Input(0)->GetMBLayout());
                BackpropToRight(*m_softmaxOfRight, Input(0)->Value(), Input(inputIndex)->Gradient(),
                    Gradient(), *m_gammaFromLattice, m_fsSmoothingWeight, m_frameDropThreshold);
                MaskMissingColumnsToZero(Input(inputIndex)->Gradient(), Input(0)->GetMBLayout(), fr);
            }
#ifdef _DEBUG
            Input(inputIndex)->InvalidateMissingGradientColumns(FrameRange(Input(inputIndex)->GetMBLayout()));
#endif
        }
        else if (inputIndex == 2)
        {
#if 1         // no gradient flows to log LLs (but otherwise we leave it to user if, e.g., another node propagates a gradient into there)
            ; // gradient does not flow here
#else
            Input(inputIndex)->SetLearningRateMultiplier(0);
            Input(inputIndex)->Gradient().SetValue(0.0); // BUGBUG: Gradients must always be added, since nodes may have multiple parents.
#endif
        }
        else
            RuntimeError("SequenceWithSoftmaxNode criterion only takes with respect to label, DNN output and log likelihood.");
    }

    static void WINAPI BackpropToLeft(const Matrix<ElemType>& logSoftmaxOfRight, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
    {
#if DUMPOUTPUT
        logSoftmaxOfRight.Print("SequenceWithSoftmaxNode Partial-logSoftmaxOfRight");
        gradientValues.Print("SequenceWithSoftmaxNode Partial-gradientValues");
        inputGradientValues.Print("SequenceWithSoftmaxNode Partial-Left-in");
#endif

        Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, gradientValues /*1x1*/, logSoftmaxOfRight, 1.0f, inputGradientValues);
#if DUMPOUTPUT
        inputGradientValues.Print("SequenceWithSoftmaxNode Partial-Left-out");
#endif
    }

    static void WINAPI BackpropToRight(const Matrix<ElemType>& softmaxOfRight, const Matrix<ElemType>& inputFunctionValues,
                                       Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues,
                                       const Matrix<ElemType>& gammaFromLattice, double hsmoothingWeight, double frameDropThresh)
    {
#if DUMPOUTPUT
        softmaxOfRight.Print("SequenceWithSoftmaxNode Partial-softmaxOfRight");
        inputFunctionValues.Print("SequenceWithSoftmaxNode Partial-inputFunctionValues");
        gradientValues.Print("SequenceWithSoftmaxNode Partial-gradientValues");
        inputGradientValues.Print("SequenceWithSoftmaxNode Partial-Right-in");
#endif

        inputGradientValues.AssignSequenceError((ElemType) hsmoothingWeight, inputFunctionValues, softmaxOfRight, gammaFromLattice, gradientValues.Get00Element());
        inputGradientValues.DropFrame(inputFunctionValues, gammaFromLattice, (ElemType) frameDropThresh);
#if DUMPOUTPUT
        inputGradientValues.Print("SequenceWithSoftmaxNode Partial-Right");
#endif
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    // -sum(left_i * log(softmax_i(right)))
    virtual void ForwardPropNonLooping()
    {
        // Initialize m_gammaCalculator
        // TODO: Would this lend itself to a unique_ptr instead of the init flag?
        if (!m_gammaCalcInitialized)
        {
            if (m_hmm.hmms.size() == 0)
            {
                LogicError("SequenceWithSoftmaxNode criterion evaluation requires HMM states to be set.");
            }
            m_gammaCalculator.init(m_hmm, m_deviceId);
            m_gammaCalcInitialized = true;
        }

        // softmax
        m_logSoftmaxOfRight->AssignLogSoftmaxOf(Input(1)->Value() /*prediction*/, true);
        m_softmaxOfRight->SetValue(*m_logSoftmaxOfRight);
        m_softmaxOfRight->InplaceExp();

        m_gammaFromLattice->SwitchToMatrixType(m_softmaxOfRight->GetMatrixType(), m_softmaxOfRight->GetFormat(), false);
        m_gammaFromLattice->Resize(*m_softmaxOfRight);
        m_gammaCalculator.calgammaformb(Value(), m_lattices, Input(2)->Value() /*log LLs*/,
                                        Input(0)->Value() /*labels*/, *m_gammaFromLattice,
                                        m_uids, m_boundaries, Input(1)->GetNumParallelSequences(),
                                        Input(0)->GetMBLayout(), m_extraUttMap, m_doReferenceAlignment);

#if NANCHECK
        Value().HasNan("SequenceWithSoftmaxNode");
#endif
#if DUMPOUTPUT
        Value().Print("SequenceWithSoftmaxNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // no layout

        if (Input(0)->OperationName() != L"InputValue" && Input(0)->OperationName() != L"SparseInputValue")
            LogicError("SequenceWithSoftmaxNode criterion requires the first input to be the label.");

        if (isFinalValidationPass)
            if (!(Input(0)->GetSampleMatrixNumRows() == Input(1)->GetSampleMatrixNumRows() && // match size
                  Input(1)->GetSampleMatrixNumRows() == Input(2)->GetSampleMatrixNumRows() &&
                  Input(0)->HasMBLayout() &&
                  Input(0)->GetMBLayout() == Input(1)->GetMBLayout())) 
            {
                LogicError("The Matrix dimension in the SequenceWithSoftmaxNode operation does not match.");
            }

        SetDims(TensorShape(1), false);

        m_gammatime = 0;
        m_partialtime = 0;
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);

        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(nodeP);

            node->m_logSoftmaxOfRight->SetValue(*m_logSoftmaxOfRight);
            node->m_softmaxOfRight->SetValue(*m_softmaxOfRight);
            node->m_gammaFromLattice->SetValue(*m_gammaFromLattice);
            node->m_fsSmoothingWeight = m_fsSmoothingWeight;
            node->m_frameDropThreshold = m_frameDropThreshold;
            node->m_doReferenceAlignment = m_doReferenceAlignment;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_logSoftmaxOfRight, matrixPool);
        RequestMatrixFromPool(m_softmaxOfRight, matrixPool);
        RequestMatrixFromPool(m_gammaFromLattice, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_logSoftmaxOfRight, matrixPool);
        ReleaseMatrixToPool(m_softmaxOfRight, matrixPool);
        ReleaseMatrixToPool(m_gammaFromLattice, matrixPool);
    }

    // TODO: method names should be CamelCase
    std::vector<shared_ptr<const msra::dbn::latticepair>>* getLatticePtr() { return &m_lattices; }
    std::vector<size_t>* getuidprt() { return &m_uids; }
    std::vector<size_t>* getboundaryprt() { return &m_boundaries; }
    std::vector<size_t>* getextrauttmap() { return &m_extraUttMap; }
    msra::asr::simplesenonehmm* gethmm() { return &m_hmm; }

    void SetSmoothWeight(double fsSmoothingWeight) { m_fsSmoothingWeight = fsSmoothingWeight; }
    void SetFrameDropThresh(double frameDropThresh) { m_frameDropThreshold = frameDropThresh; }
    void SetReferenceAlign(const bool doreferencealign) { m_doReferenceAlignment = doreferencealign; }

    void SetGammarCalculationParam(const double& amf, const double& lmf, const double& wp, const double& bMMIfactor, const bool& sMBR)
    {
        msra::lattices::SeqGammarCalParam param;
        param.amf = amf;
        param.lmf = lmf;
        param.wp = wp;
        param.bMMIfactor = bMMIfactor;
        param.sMBRmode = sMBR;
        m_gammaCalculator.SetGammarCalculationParams(param);
    }

    void gettime(unsigned long long& gammatime, unsigned long long& partialtime)
    {
        gammatime = m_gammatime;
        partialtime = m_partialtime;
    }

protected:
    shared_ptr<Matrix<ElemType>> m_logSoftmaxOfRight;
    shared_ptr<Matrix<ElemType>> m_softmaxOfRight;
    shared_ptr<Matrix<ElemType>> m_gammaFromLattice;
    bool m_invalidMinibatch; // for single minibatch
    double m_frameDropThreshold;
    double m_fsSmoothingWeight; // frame-sequence criterion interpolation weight    --TODO: can this be done outside?
    double m_seqGammarAMF;
    double m_seqGammarLMF;
    double m_seqGammarWP;
    double m_seqGammarbMMIFactor;
    bool m_seqGammarUsesMBR;
    bool m_doReferenceAlignment;
    std::vector<shared_ptr<const msra::dbn::latticepair>> m_lattices;
    msra::asr::simplesenonehmm m_hmm;
    msra::lattices::GammaCalculation<ElemType> m_gammaCalculator;
    bool m_gammaCalcInitialized;
    std::vector<size_t> m_uids;
    std::vector<size_t> m_boundaries;
    std::vector<size_t> m_extraUttMap;

    unsigned long long m_gammatime; // TODO: what are these? Not even the context can be guessed from these names.
    unsigned long long m_partialtime;
};

template class SequenceWithSoftmaxNode<float>;
template class SequenceWithSoftmaxNode<double>;

// -----------------------------------------------------------------------
// LatticeSequenceWithSoftmaxNode (label, prediction, loglikelihood, lattice)
// Similar to the SequenceWithSoftmaxNode, but is using the new deserializer.
//
// -----------------------------------------------------------------------

template <class ElemType>
class LatticeSequenceWithSoftmaxNode : public SequenceWithSoftmaxNode<ElemType>, public NumInputs<4>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"LatticeSequenceWithSoftmax";
    }

public:
    LatticeSequenceWithSoftmaxNode(DEVICEID_TYPE deviceId, const std::wstring& name, const std::wstring& symListPath, const std::wstring& phonePath, const std::wstring& stateListPath, const std::wstring& transProbPath, const std::wstring& latticeConfigPath,
        float hSmoothingWeight, float frameDropThresh, bool doReferenceAlign, bool seqGammarUsesMBR, float seqGammarAMF, float seqGammarLMF, float seqGammarBMMIFactor, float seqGammarWordPen)
        : SequenceWithSoftmaxNode<ElemType>(deviceId, name), m_symListPath(symListPath), m_phonePath(phonePath), m_stateListPath(stateListPath), m_transProbPath(transProbPath), m_latticeConfigPath(latticeConfigPath)
    {
        if (sizeof(ElemType) != sizeof(float))
            LogicError("LatticeSequenceWithSoftmaxNode currently only supports floats.\n"); // due to the binary reader restrictions 

        if (symListPath.size() == 0 || phonePath.size() == 0 || stateListPath.size() == 0 || transProbPath.size() == 0)
            LogicError("Ensure that symListPath, phonePath, stateListPath and transProbPath parameters are specified.\n");

        if (doReferenceAlign)
            LogicError("SE training with alignment is currently not supported.\n");

        LoadConfigsFromFile();

        InitSEParams(m_symListPath, m_phonePath, m_stateListPath, m_transProbPath);
        this->m_fsSmoothingWeight = hSmoothingWeight;
        this->m_frameDropThreshold = frameDropThresh;
        this->m_doReferenceAlignment = doReferenceAlign;
        this->m_seqGammarUsesMBR = seqGammarUsesMBR;
        this->m_seqGammarAMF = seqGammarAMF;
        this->m_seqGammarLMF = seqGammarLMF;
        this->m_seqGammarbMMIFactor = seqGammarBMMIFactor;
        this->m_seqGammarWP = seqGammarWordPen;

        this->SetGammarCalculationParam(seqGammarAMF, seqGammarLMF, seqGammarWordPen, seqGammarBMMIFactor, seqGammarUsesMBR);
    }

    LatticeSequenceWithSoftmaxNode(DEVICEID_TYPE deviceId, const std::wstring& name)
        : SequenceWithSoftmaxNode<ElemType>(deviceId, name)
    {
    }

    LatticeSequenceWithSoftmaxNode(const ScriptableObjects::IConfigRecordPtr configp)
        : LatticeSequenceWithSoftmaxNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"symListPath"), configp->Get(L"phonePath"), configp->Get(L"stateListPath"), configp->Get(L"transProbPath"), configp->Get(L"latticeConfigPath"),
            configp->Get(L"hSmoothingWeight"), configp->Get(L"frameDropThresh"), configp->Get(L"doReferenceAlign"), configp->Get(L"seqGammarUsesMBR"), configp->Get(L"seqGammarAMF"), configp->Get(L"seqGammarLMF"), configp->Get(L"seqGammarBMMIFactor"), configp->Get(L"seqGammarWordPen")
        )
    {
        AttachInputsFromConfig(configp, 4);
    }

    // compute gradients to input observations, the weights to the observations, and the class log posterior probabilities
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        SequenceWithSoftmaxNode<ElemType>::BackpropToNonLooping(inputIndex);
    }

    // -sum(left_i * log(softmax_i(right)))
    virtual void ForwardPropNonLooping()
    {
        this->m_lattices.clear();
        this->m_uids.clear();
        this->m_boundaries.clear();
        this->m_extraUttMap.clear();
        this->m_invalidMinibatch = false;

        if (InputRef(3).ValuePtrRef()->GetDeviceId() != CPUDEVICE)
            LogicError("Due to their size, lattices should be allocated on CPU memory");

        const char* bufferStart = reinterpret_cast<char*>(InputRef(3).ValuePtrRef()->Data());

        let& labelMBLayout = InputRef(0).GetMBLayout();
        const auto& labelSequences = labelMBLayout->GetAllSequences();

        let& latticeMBLayout = InputRef(3).GetMBLayout();
        size_t latticeMBNumTimeSteps = latticeMBLayout->GetNumTimeSteps();

        InputRef(0).ValuePtrRef()->VectorMax(*m_maxIndexes, *m_maxValues, true);
        vector<size_t> labelSequencesMap;
        for (size_t i = 0; i < labelSequences.size(); i++)
        {
            if (labelSequences[i].seqId == GAP_SEQUENCE_ID)
                continue;
            labelSequencesMap.push_back(labelSequences[i].seqId);
            auto& currentLabelSeq = labelSequences[i];

            // Fill up labels
            auto columnIndices = labelMBLayout->GetColumnIndices(currentLabelSeq);

            for (size_t ci = 0; ci < columnIndices.size(); ci++)
            {
                size_t refId = (int)(*m_maxIndexes)(0, columnIndices[ci]);
                this->m_uids.push_back(refId);
            }
            this->m_extraUttMap.push_back(labelSequences[i].s);
        }

        this->m_lattices.resize(labelSequencesMap.size());
        try {
#pragma omp parallel for 
            for (long i = 0; i < labelSequences.size(); i++)
            {
                if (labelSequences[i].seqId == GAP_SEQUENCE_ID)
                    continue;

                auto& currentLabelSeq = labelSequences[i];

                // Fill up lattice
                auto& currentLatticeSeq = latticeMBLayout->FindSequence(currentLabelSeq.seqId);
                std::shared_ptr<msra::dbn::latticepair> latticePair(new msra::dbn::latticepair);
                const char* buffer = bufferStart + latticeMBNumTimeSteps * sizeof(float) * currentLatticeSeq.s + currentLatticeSeq.tBegin;
                latticePair->second.ReadFromBuffer(buffer, m_idmap, m_idmap.back());
                assert((currentLabelSeq.tEnd - currentLabelSeq.tBegin) == latticePair->second.info.numframes);
                // The size of the vector is small -- the number of sequences in the minibatch. 
                // Iteration likely will be faster than the overhead with unordered_map
                for (size_t pos = 0; pos < labelSequencesMap.size();pos++)
                {
                    if (labelSequencesMap[pos] == labelSequences[i].seqId)
                    {
                        this->m_lattices[pos] = latticePair;
                        break;
                    }
                }
            }
        }
        catch (...)
        {
            fprintf(stderr, "WARNING: Failed to parse lattice. Skipping minibatch...\n");
            this->m_invalidMinibatch = true;
        }

        if (!this->m_invalidMinibatch)
        {
            this->m_boundaries.resize(this->m_uids.size());
            std::fill(this->m_boundaries.begin(), this->m_boundaries.end(), 0);
            SequenceWithSoftmaxNode<ElemType>::ForwardPropNonLooping();
        }
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_symListPath;
        fstream << m_phonePath;
        fstream << m_stateListPath;
        fstream << m_transProbPath;
        fstream << m_latticeConfigPath;
        fstream << this->m_frameDropThreshold;
        fstream << this->m_fsSmoothingWeight;
        fstream << this->m_seqGammarAMF;
        fstream << this->m_seqGammarLMF;
        fstream << this->m_seqGammarWP;
        fstream << this->m_seqGammarbMMIFactor;
        fstream << this->m_seqGammarUsesMBR;
        fstream << this->m_doReferenceAlignment;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_symListPath;
        fstream >> m_phonePath;
        fstream >> m_stateListPath;
        fstream >> m_transProbPath;
        fstream >> m_latticeConfigPath;
        fstream >> this->m_frameDropThreshold;
        fstream >> this->m_fsSmoothingWeight;
        fstream >> this->m_seqGammarAMF;
        fstream >> this->m_seqGammarLMF;
        fstream >> this->m_seqGammarWP;
        fstream >> this->m_seqGammarbMMIFactor;
        fstream >> this->m_seqGammarUsesMBR;
        fstream >> this->m_doReferenceAlignment;
        try
        {
            LoadConfigsFromFile();
            InitSEParams(m_symListPath, m_phonePath, m_stateListPath, m_transProbPath);
            this->SetGammarCalculationParam(this->m_seqGammarAMF, this->m_seqGammarLMF, this->m_seqGammarWP, this->m_seqGammarbMMIFactor, this->m_seqGammarUsesMBR);
        }
        catch (...)
        {
            fprintf(stderr, "WARNING: Failed to open one or more of the files.");
        }
    }

    void LoadConfigsFromFile()
    {
        // Workaround for loading a trained model from a different location
        std::string latticeConfigPathStr = Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(m_latticeConfigPath));
        wifstream file(latticeConfigPathStr.c_str());
        if (file.good())
        {
            wstring str;
            getline(file, str);
            m_symListPath = str;
            getline(file, str);
            m_phonePath = str;
            getline(file, str);
            m_stateListPath = str;
            getline(file, str);
            m_transProbPath = str;
        }
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        SequenceWithSoftmaxNode<ElemType>::Validate(isFinalValidationPass);

        if (isFinalValidationPass)
        {
            // Make sure lattices are pre allocated on CPU, due to their size.
            Input(3)->ValuePtrRef()->TransferToDeviceIfNotThere(CPUDEVICE, true /*moving completely*/, true /*preserving no data*/);
        }
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        SequenceWithSoftmaxNode<ElemType>::CopyTo(nodeP, newName, flags);

        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<LatticeSequenceWithSoftmaxNode<ElemType>>(nodeP);

            if (node) 
            {
                node->m_idmap = m_idmap;
                node->m_symListPath = m_symListPath;
                node->m_phonePath = m_phonePath;
                node->m_stateListPath = m_stateListPath;
                node->m_stateListPath = m_transProbPath;
            }
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        SequenceWithSoftmaxNode<ElemType>::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maxIndexes, matrixPool);
        RequestMatrixFromPool(m_maxValues, matrixPool);
    }

private: 
    msra::lattices::archive::symbolidmapping m_idmap;
    std::wstring m_symListPath;
    std::wstring m_phonePath;
    std::wstring m_stateListPath;
    std::wstring m_transProbPath;
    std::wstring m_latticeConfigPath;
    shared_ptr<Matrix<ElemType>> m_maxIndexes, m_maxValues;

    void InitSEParams(const std::wstring& symListPath, const std::wstring& phonePath, const std::wstring& stateListPath, const std::wstring& transProbPath) 
    {
        LOGPRINTF(stderr, "Reading files\n %ls \n %ls \n %ls \n %ls \n", symListPath.c_str(), phonePath.c_str(), stateListPath.c_str(), transProbPath.c_str());
        this->m_hmm.loadfromfile(phonePath, stateListPath, transProbPath);
        auto symmap = this->m_hmm.getsymmap();
        msra::lattices::archive::GetSymList(m_idmap, symListPath, symmap);
    }
};

template class LatticeSequenceWithSoftmaxNode<float>;
template class LatticeSequenceWithSoftmaxNode<double>;

// -----------------------------------------------------------------------
// DummyCriterionNode (objectiveValues, userSuppliedGradient, prediction)
// TODO: Rename to CustomCriterionNode?
//
// Apply user-supplied gradient, computed as Forward(), as the gradient into 'prediction'.
//
// predictionsGradient += userSuppliedGradient * scalarGradientFromTop
//
// This training criterion node allows to compute objectives and gradient
// with custom CNTK expressions (as Forward() computations). It has 3 inputs:
// 1. custom objective values to be summed up and passed up
// 2. custom gradient values to be passed down as the gradient into 'prediction'
// 3. prediction: the node to pass the custom gradient into
// -----------------------------------------------------------------------

template <class ElemType>
class DummyCriterionNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"DummyCriterion";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(DummyCriterionNode);
    DummyCriterionNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        if (inputIndex == 0)
            LogicError("DummyCriterionNode: Gradients with respect to objective features are not necessary, not implemented.\n");
        else if (inputIndex == 1)
            LogicError("DummyCriterionNode: Gradients with respect to derivative features are not necessary, not implemented.\n");
        else if (inputIndex == 2)
        {
            // predictionsGradient += userSuppliedGradient * scalarGradientFromTop
            auto gradient = Input(2)->GradientFor(fr);
            Matrix<ElemType>::Multiply1x1AndWeightedAdd(+1.0f, /*gradient from top:*/Gradient() /*1x1*/, /*user-supplied gradient:*/Input(1)->ValueFor(fr), 1.0f, /*add to:*/gradient);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        Value().VerifySize(1, 1);
        assert(Input(0)->Value().GetNumRows() == 1);
        Value().SetValue(Input(0)->Value().SumOfElements());
#if NANCHECK
        Value().HasNan("DummyCriterionNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (Input(0)->OperationName() != L"InputValue")
            LogicError("DummyCriterionNode criterion requires the first input to be computed objectives.");
        if (Input(1)->OperationName() != L"InputValue")
            LogicError("DummyCriterionNode criterion requires the second input to be computed derivatives.");
        if (isFinalValidationPass)
        {
            if (Input(0)->GetSampleMatrixNumRows() == 0
                || Input(1)->GetSampleMatrixNumRows() == 0
                || Input(2)->GetSampleMatrixNumRows() == 0)
                LogicError("DummyCriterionNode operation: one of the operands has 0 elements.");
            if (Input(1)->GetSampleMatrixNumRows() != Input(2)->GetSampleMatrixNumRows()
                || Input(0)->GetSampleMatrixNumCols() != Input(2)->GetSampleMatrixNumCols()
                || Input(1)->GetSampleMatrixNumCols() != Input(2)->GetSampleMatrixNumCols())
                LogicError("The Matrix dimension in the DummyCriterionNode operation does not match.");
        }

        SetDims(TensorShape(1), false);
    }
};

template class DummyCriterionNode<float>;
template class DummyCriterionNode<double>;

// -----------------------------------------------------------------------
// ForwardBackwardNode (graph, prediction, delayConstraint)
// CTC training criterion, primarily based on the paper "Connectionist Temporal Classification: Labelling Unsegmented
// Sequence Data with Recurrent Neural Networks", ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
// blankTokenId (input): id of the blank token. If specified as SIZE_MAX, will be replaced with (numberOfLabels - 1)
// delayConstraint -- label output delay constraint introduced during training that allows to have shorter delay during inference. 
//      This using the original time information to enforce that CTC tokens only get aligned within a time margin.
//      Setting this parameter smaller will result in shorter delay between label output during decoding, yet may hurt accuracy.
//      delayConstraint=-1 means no constraint
// -----------------------------------------------------------------------

template<class ElemType>
class ForwardBackwardNode : public  ComputationNodeNonLooping<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"ForwardBackward";
    }
public:
    ForwardBackwardNode(DEVICEID_TYPE deviceId, const wstring & name, size_t blankTokenId=SIZE_MAX, int delayConstraint=-1) :
        Base(deviceId, name), m_blankTokenId(blankTokenId), m_delayConstraint(delayConstraint)
    {
    }

    ForwardBackwardNode(const ScriptableObjects::IConfigRecordPtr configp)
        : ForwardBackwardNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"blankTokenId"), configp->Get(L"delayConstraint"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    // Compute gradients to input observations, the weights to the observations, and the class log posterior probabilities
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        // Left node must be a scalar
        if (inputIndex == 0)  //left derivative
        {
            BackpropToLeft(*m_logSoftmaxOfRight, InputRef(inputIndex).Gradient(), Gradient());
        }
        else if (inputIndex == 1)
        {
            FrameRange frameRange(InputRef(0).GetMBLayout());
            BackpropToRight(*m_softmaxOfRight, InputRef(inputIndex).Gradient(), Gradient(), *m_CTCposterior);
            InputRef(inputIndex).MaskMissingGradientColumnsToZero(frameRange);
        }
        else
           RuntimeError("ForwardBackwardNode criterion expects only two inputs: labels and network output.");
    }

    void BackpropToLeft(const Matrix<ElemType>& logSoftmaxOfRight, Matrix<ElemType>& inputGradientValues,
        const Matrix<ElemType>& gradientValues)
    {
#if DUMPOUTPUT
        logSoftmaxOfRight.Print("ForwardBackwardNode Partial-logSoftmaxOfRight");
        gradientValues.Print("ForwardBackwardNode Partial-gradientValues");
        inputGradientValues.Print("ForwardBackwardNode Partial-Left-in");
#endif

        Matrix<ElemType>::ScaleAndAdd(-gradientValues.Get00Element(), logSoftmaxOfRight, inputGradientValues);

#if DUMPOUTPUT
        inputGradientValues.Print("ForwardBackwardNode Partial-Left-out");
#endif
    }

    void BackpropToRight(const Matrix<ElemType>& softmaxOfRight, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues,
        const Matrix<ElemType> &CTCposterior)
    {
#if DUMPOUTPUT
        softmaxOfRight.Print("ForwardBackwardNode Partial-softmaxOfRight");
        inputFunctionValues.Print("ForwardBackwardNode Partial-inputFunctionValues");
        gradientValues.Print("ForwardBackwardNode Partial-gradientValues");
        inputGradientValues.Print("ForwardBackwardNode Partial-Right-in");
#endif  
        // inputGradientValues+= gradientValues*(softmaxOfRight - CTCposterior)
        Matrix<ElemType>::AddScaledDifference(gradientValues, softmaxOfRight, CTCposterior, inputGradientValues); 

#if DUMPOUTPUT
        inputGradientValues.Print("ForwardBackwardNode Partial-Right");
#endif
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void ForwardPropNonLooping() override
    {
        m_logSoftmaxOfRight->AssignLogSoftmaxOf(InputRef(1).Value(), true);
        m_softmaxOfRight->SetValue(*m_logSoftmaxOfRight);
        m_softmaxOfRight->InplaceExp();

        m_CTCposterior->SwitchToMatrixType(m_softmaxOfRight->GetMatrixType(), m_softmaxOfRight->GetFormat(), false);
        m_CTCposterior->Resize(m_softmaxOfRight->GetNumRows(), m_softmaxOfRight->GetNumCols());

        FrameRange fr(InputRef(0).GetMBLayout());
        InputRef(0).ValueFor(fr).VectorMax(*m_maxIndexes, *m_maxValues, true);
        // compute CTC score
        m_GammaCal.doCTC(Value(), *m_logSoftmaxOfRight, *m_maxIndexes, *m_maxValues, *m_CTCposterior, InputRef(0).GetMBLayout(), m_blankTokenId, m_delayConstraint);

#if NANCHECK
        functionValues.HasNan("ForwardBackwardNode");
#endif
#if DUMPOUTPUT
        functionValues.Print("ForwardBackwardNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // no layout

        if (isFinalValidationPass) 
        {
            if (!(Input(0)->GetSampleMatrixNumRows() == Input(1)->GetSampleMatrixNumRows() && // match vector dimension
                Input(0)->HasMBLayout() &&
                Input(0)->GetMBLayout() == Input(1)->GetMBLayout()))
            {
                LogicError("The Matrix dimension in the ForwardBackwardNode operation does not match.");
            }

            auto leftNode = dynamic_pointer_cast<LabelsToGraphNode<ElemType>>(Input(0));
            if (!leftNode)
                LogicError("ForwardBackwardNode: Please pass LabelsToGraph(labels) for second argument");
        }

        SetDims(TensorShape::Scalar(Environment().IsV2Library()), false);
    }

    virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ForwardBackwardNode<ElemType>>(nodeP);

            node->m_logSoftmaxOfRight->SetValue(*m_logSoftmaxOfRight);
            node->m_softmaxOfRight->SetValue(*m_softmaxOfRight);
            node->m_CTCposterior->SetValue(*m_CTCposterior);
            node->m_maxIndexes->SetValue(*m_maxIndexes);
            node->m_maxValues->SetValue(*m_maxValues);
            node->m_delayConstraint = m_delayConstraint;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_logSoftmaxOfRight, matrixPool);
        RequestMatrixFromPool(m_softmaxOfRight, matrixPool);
        RequestMatrixFromPool(m_CTCposterior, matrixPool);
        RequestMatrixFromPool(m_maxIndexes, matrixPool);
        RequestMatrixFromPool(m_maxValues, matrixPool);
    }

    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_logSoftmaxOfRight, matrixPool);
        ReleaseMatrixToPool(m_softmaxOfRight, matrixPool);
        ReleaseMatrixToPool(m_CTCposterior, matrixPool);
        ReleaseMatrixToPool(m_maxIndexes, matrixPool);
        ReleaseMatrixToPool(m_maxValues, matrixPool);
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();

        size_t cols = Input(0)->Value().GetNumCols();
        m_maxIndexes->Resize(1, cols);
        m_maxValues->Resize(1, cols);
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_delayConstraint;
        fstream << m_blankTokenId;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_delayConstraint;
        fstream >> m_blankTokenId;
    }

    int DelayConstraint() { return m_delayConstraint; }
    size_t BlankTokenId() { return m_blankTokenId; }

protected:
    virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() { return true; }
    shared_ptr<Matrix<ElemType>> m_logSoftmaxOfRight;
    shared_ptr<Matrix<ElemType>> m_softmaxOfRight;
    shared_ptr<Matrix<ElemType>> m_CTCposterior;
    shared_ptr<Matrix<ElemType>> m_maxIndexes;
    shared_ptr<Matrix<ElemType>> m_maxValues;

    msra::lattices::GammaCalculation<ElemType> m_GammaCal;
    size_t m_blankTokenId;
    int m_delayConstraint;
};

template class ForwardBackwardNode<float>;
template class ForwardBackwardNode<double>;

// -----------------------------------------------------------------------
// StopGradientNode (Input)
// Outputs its input as it and prevents any gradient contribution from its output to its input.
// TODO: This could be more easily implemented as a unary operation, like PassNode.
// -----------------------------------------------------------------------
template <class ElemType>
class StopGradientNode : public UnaryElementWiseNode<ElemType>
{
    typedef UnaryElementWiseNode<ElemType> Base; 
    UsingUnaryElementwiseNodeBaseMembers;
    static const std::wstring TypeName() { return L"StopGradient"; }
public:
    DeclareConstructorFromConfigWithNumInputs(StopGradientNode);
    StopGradientNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        auto result = ValueFor(fr);
        auto inputValue = InputRef(0).ValueFor(fr);
        // TODO:@Amit Due to current limitation of the network builder, we can't bypass the memory copy operation at this step. 
        // But idealy, we should just pass the value of input as this node's output
        result.AssignValuesOf(inputValue);
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        // Do nothing to short circuit the gradient backward propagation
        // Update: Short circuit from Backprop now, so we shouldn't reach here.
        RuntimeError("Unexpected method called in StopGradientNode: BackpropTo. ");
    }

    void Backprop(const FrameRange& fr, bool childrenInThisLoop, bool childrenInOuterLoop) override
    {
        // Do nothing to short circuit the gradient backward propagation
        // In Base(ComputationNode), Backprop validates m_needsgradient == true if any child needs gradient, and calls BackpropTo. We can short circuit this process altogether. 
        // In current implementation we set m_needsgradient = false for StopGradientNode, so that we can pass validate check from nodes that don't 
        // support input with gradient (e.g. ToSequenceNode does not support gradient propgation to its Input(1) denoting sequence lengths), 
        // as well as short circuit some unnecessary gradient backprop. 
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }
};

template class StopGradientNode<float>;
template class StopGradientNode<double>;

// -----------------------------------------------------------------------
// AssignNode (RefInput, Input)
// -----------------------------------------------------------------------
template <class ElemType>
class AssignNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"Assign"; }

    shared_ptr<Matrix<ElemType>> m_result;

public:
    DeclareConstructorFromConfigWithNumInputs(AssignNode);
    AssignNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_result->Resize(Input(0)->Value());
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        auto& result = Value();
        auto& inputValue = InputRef(1).Value();

        if (inputValue.GetNumElements() != result.GetNumElements())
        {
            InvalidArgument("%ls %ls operation: unexpected dimension mismatch", NodeName().c_str(), OperationName().c_str());
        }

        m_result->AssignValuesOf(inputValue);
        result.AssignValuesOf(inputValue);
    }

    virtual void /*ComputationNodeNonLooping::*/ PostForwardAndBackProp() override
    {
        auto& refValue = InputRef(0).Value();
        refValue.AssignValuesOf(*m_result);

        // We update Input(0) so bump the timestamp for the new data.
        Input(0)->BumpEvalTimeStamp();
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        if (inputIndex == 1)
            Input(1)->Gradient() += Gradient();
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryZip(isFinalValidationPass, false);

        if (Input(0)->HasMBLayout() || Input(1)->HasMBLayout())
            InvalidArgument("AssignNode: None of the inputs can have dynamic axes.");
        //only check layout in final pass, as there may be free dimension axis
        if (isFinalValidationPass && Input(0)->GetSampleLayout() != Input(1)->GetSampleLayout())
            InvalidArgument("AssignNode: All inputs should have same sample layout.");
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_result, matrixPool);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }
};

template class AssignNode<float>;
template class AssignNode<double>;

// -----------------------------------------------------------------------
// OutputMultiplexerNode(userDefinedV2FunctionNode, outputIndex)
// ComputationNode for selecting one of the multiple outputs of UserDefinedV2FunctionNode
// This is needed since the CNTK computation engin natively does not support
// nodes with multiple outputs and hence, we need a separate node to multiplex 
// the additional outputs.
// -----------------------------------------------------------------------

// TODO: We currently only support external nodes that cannot be part of CNTK recurrent loops
template <class ElemType>
class OutputMultiplexerNode final : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"OutputMultiplexer"; }

public:
    OutputMultiplexerNode(DEVICEID_TYPE deviceId, const wstring& name, size_t outputIndex = 0)
        : Base(deviceId, name), m_outputIndex(outputIndex)
    {
        if (outputIndex == 0)
            LogicError("OutputMultiplexerNode ctor must not be instantiated with outputIndex == 0");
    }

    virtual void ForwardPropNonLooping() override
    {
        // TODO: We should avoid this copy but that requires carefully managing the 
        // lifetimes of the Value objects since to be able to directly use the 
        // input Value as its output, we have to make sure that the input's Value
        // is not reused until all dependents of this node are finished.
        auto inputNode = Input(0)->template As<MultiOutputNode<ElemType>>();
        Value().AssignValuesOf(*inputNode->m_outputsValue[m_outputIndex]);
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        // TODO: We should avoid this copy but that requires carefully managing the 
        // lifetimes of the Gradient objects since to be able to directly use the 
        // Gradient as input's gradient, we have to make sure that the Gradient
        // is not reused until all the inputs are finished backpropagating to their inputs.
        auto inputNode = Input(0)->template As<MultiOutputNode<ElemType>>();
        inputNode->m_outputsGradient[m_outputIndex]->SetValue(Gradient());
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        auto inputNode = Input(0)->template As<MultiOutputNode<ElemType>>();
        m_pMBLayout = inputNode->m_outputsMBLayout[m_outputIndex];
        SetDims(inputNode->m_outputsShape[m_outputIndex], HasMBLayout());
    }

private:
    size_t m_outputIndex;
};

template class OutputMultiplexerNode<float>;
template class OutputMultiplexerNode<double>;

// -----------------------------------------------------------------------
// CustomProxyOpNode is a placeholder node for a quantized operations.
// It enables saving a model with its parameters so that they can be loaded
// from the optimized implementation (Halide) for execution.
// -----------------------------------------------------------------------

template <class ElemType>
class CustomProxyOpNode : public ComputationNode<ElemType> /* Not deriving from NumInputs, public NumInputs<4>*/
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"CustomProxyOpNode"; }

public:
    CustomProxyOpNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    CustomProxyOpNode(const ScriptableObjects::IConfigRecordPtr configp)
        : CustomProxyOpNode(configp->Get(L"deviceId"), L"<placeholder>")
    {
        AttachInputsFromConfig(configp);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        NOT_IMPLEMENTED
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        NOT_IMPLEMENTED
    }
};

template class CustomProxyOpNode<float>;

} } }
