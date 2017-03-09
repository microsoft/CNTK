//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "ScriptableObjects.h"
#include "TensorShape.h"
#include "Matrix.h"

#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

static const wchar_t* ConstantInitializerTypeName =         L"constant";
static const wchar_t* UniformBSInitializerTypeName =        L"uniform";     // for legacy reason, "uniform" is taken in BrainScript to represent uniform distribution [-0.05, 0.05]
static const wchar_t* UniformInitializerTypeName =          L"uniform1";
static const wchar_t* GaussianInitializerTypeName =         L"gaussian";    // legacy for BrainScript normal distribution pre-scaled by sqrt(0.04 / fanin)
static const wchar_t* NormalInitializerTypeName =           L"normal";
static const wchar_t* XavierInitializerTypeName =           L"xavier";
static const wchar_t* GlorotUniformInitializerTypeName =    L"glorotUniform";
static const wchar_t* GlorotNormalInitializerTypeName =     L"glorotNormal";
static const wchar_t* HeUniformInitializerTypeName =        L"heUniform";
static const wchar_t* HeNormalInitializerTypeName =         L"heNormal";
static const wchar_t* BilinearInitializerTypeName =         L"bilinear";

// -----------------------------------------------------------------------
// LearnableParameter (/*no input*/)
// represents weight matrices and biases
// TODO: add -Node to the class name
// -----------------------------------------------------------------------

template <class ElemType>
class LearnableParameter : public ComputationNode<ElemType>, public NumInputs<0>, public IFreezable, public TransformerNode
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"LearnableParameter"; }

    void InitShape(const TensorShape& shape);

public:
    // this constructor is always run (all other constructors call this one)
    LearnableParameter(DEVICEID_TYPE deviceId, const wstring& name) :
        Base(deviceId, name)
    {
        SetLearningRateMultiplier(1.0f); // enable normal learning by default
        MarkValueNonSharable();
        m_initString = L"fromValue"; // default init is with 0; typically overwritten
        m_initValue = 0;
        m_regMultiplier = 1.0f; // enable reg in update by default
    }
    LearnableParameter(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& shape) :
        LearnableParameter(deviceId, name)
    {
        InitShape(shape);
        LazyInitParameters();
    }
    LearnableParameter(DEVICEID_TYPE deviceId, const wstring& name, size_t rows, size_t cols) :
        LearnableParameter(deviceId, name, TensorShape(rows, cols))
    {
    }
    LearnableParameter(const ScriptableObjects::IConfigRecordPtr configp);

    // initialize after plain constructor; for use by NDL
    void PostInitParameters(const std::wstring& initString, // "uniform"|"gaussian"|"fixedValue"
                            ElemType initValue,             //  scale   | scale    | value
                            unsigned long randomSeed = 0,
                            bool initOnCPUOnly = false);

    // Initialize with bilinear interpolation coefficients (useful for deconvolution layer).
    void InitBilinear(size_t kernelWidth, size_t kernelHeight)
    {
        InitBilinear(Value(), GetSampleLayout(), kernelWidth, kernelHeight, m_deviceId);
    }

    // initialize by reading a matrix from a text file
    void InitFromFile(const std::wstring& initFromFilePath);

public:
    // initialize with random numbers
    // If 'initOnCPUOnly' then always init on CPU, making initialization consistent across both (for testing).
    static std::tuple<size_t, size_t, ElemType> InitRandom(Matrix<ElemType>& valueMatrix,
                                                           const TensorShape& sampleShape,
                                                           const std::wstring& type,
                                                           const unsigned long randomSeed,
                                                           const ElemType initValueScale,
                                                           const size_t initFilterRank,
                                                           const int initOutputRank,
                                                           const bool initOnCPUOnly,
                                                           DEVICEID_TYPE deviceId);

    static void InitBilinear(Matrix<ElemType>& valueMatrix, const TensorShape& sampleShape, size_t kernelWidth, size_t kernelHeight, DEVICEID_TYPE deviceId);

private:
    void InitRandom(const std::wstring& type, const unsigned long randomSeed, const ElemType initValueScale, const size_t initFilterRank, const int initOutputRank, const bool initOnCPUOnly)
    {
        size_t fanOut, fanIn;
        ElemType range;
        std::tie(fanOut, fanIn, range) = InitRandom(Value(), GetSampleLayout(), type, randomSeed, initValueScale, initFilterRank, initOutputRank, initOnCPUOnly, m_deviceId);
        if (fanOut == 0) // Shape not yet initialized
            return;

        bool log = GetEnvironmentPtr() && Environment().traceLevel > 0; // note: this will not log before node is part of network
        if (log)
        {
            fprintf(stderr, "%ls: Initializing Parameter[%s] <- %ls(seed=%d, init dims=[%d x %d], range=%f(%f*%f), onCPU=%s.\n)",
                    NodeDescription().c_str(), string(GetSampleLayout()).c_str(), m_initString.c_str(),
                    (int)randomSeed, (int)fanOut, (int)fanIn, range, range/initValueScale, initValueScale, initOnCPUOnly ? "true" : "false");
        }
    }

    // helper to initialize from a matrix read from a text file or a string literal
    void InitFromArray(const std::vector<ElemType>& array, size_t numRows, size_t numCols);

    // deferred initialization
    void LazyInitParameters();

public:
    // reload parameters from file
    // This is called from MEL.
    void ReviseFromFile(const std::wstring& reviseFromFilePath);

    virtual void Save(File& fstream) const override;
    virtual void Load(File& fstream, size_t modelVersion) override;

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override;

    // computation functions don't do anything for parameter nodes
    virtual void UpdateFunctionMBSize() override;
    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange&) override;
    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&) override;
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;

    // called from ComputationNode::ValidateInferInputDimsFrom()
    // In case of an error, this function just backs out without updating.
    // The caller must verify the dimensions.
    // This is a bit weird since it is called after this node has been Validated once.
    // BUGBUG: This will clear out any random initialization to 0. So currently this is not usable for most cases.
    void InferInputDimsFrom(const TensorShape& otherShape);

    virtual void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override;

    // called from CloneFunction(..., parameters="constant")
    virtual void FreezeParameters() override; // from IFreezable

    // Setting the reg multiplier for a learnable node, effecting L1Reg and L2Reg both.
    void SetRegMultiplier(float regMultiplier)
    {
        m_regMultiplier = regMultiplier;
    }
    // called from SGD UpdateWeights, to adjust the reg for each node
    float GetRegMultiplier() const { return m_regMultiplier; }

    virtual bool /*TransformerNode::*/SupportsTransformOnInput(size_t /*index*/) override
    {
        RuntimeError("LearnableParameter should not be asked for input transforms, since it has no inputs.");
    }

    // Has no inputs, hence does not support transforms on any input.
    virtual void /*TransformerNode::*/ComputeTransforms() override
    {
        RuntimeError("LearnableParameter should not be asked for input transforms, since it has no inputs.");
    }

private:
    // init parameters for deferred initialization (which happens in Validate())
    std::wstring m_initString; // if non-empty then deferred initialization is needed. Gets cleared upon completion of deferred init.
    unsigned long m_randomSeed;
    ElemType m_initValueScale;
    size_t m_initFilterRank;
    int m_initOutputRank;
    bool m_initOnCPUOnly;
    ElemType m_initValue;

    // flags related to gradient update
    float m_regMultiplier; // The multiplier to adjust the L1Reg and L2Reg for Learnable node
};

// -----------------------------------------------------------------------
// DynamicAxisNode (/*no input*/)
// This is a holder for MBLayout objects shared across inputs.
// -----------------------------------------------------------------------
template <class ElemType>
class DynamicAxisNode : public ComputationNode<ElemType>, public NumInputs<0>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"DynamicAxis"; }
public:
    DynamicAxisNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
        // BUGBUG: In BS, the node name is not known during node instantiation.
        // This may require to pass the display name as a separate parameter.

        // This is the whole point of this class: Introduce a new MBLayout that others can use.
        LinkToMBLayout(make_shared<MBLayout>(1, 0, name));
        // We need some shape, or validation fails.
        SetDims(TensorShape(1,1), true);
    }
    DynamicAxisNode(const ScriptableObjects::IConfigRecordPtr configp)
        : DynamicAxisNode(configp->Get(L"deviceId"), L"<placeholder>")
    {
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange&) override
    {
        RuntimeError("%ls is a special node only to be used as input to the Input() node.", NodeDescription().c_str());
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&)
    {
        LogicError("%ls is a leaf node. BackpropTo() should never be called.", NodeDescription().c_str());
    }
};

template class DynamicAxisNode<float>;
template class DynamicAxisNode<double>;


// -----------------------------------------------------------------------
// InputValueBase (/*no input*/)
// Base class for InputValue and SparseInputValue (typically fed by a DataReader)
// this covers four types: (regular vs. image) x (non-sparse vs. sparse)
// TODO: add -Node to the class names
// -----------------------------------------------------------------------

template <class ElemType>
class InputValueBase : public ComputationNode<ElemType>, public NumInputs<0>, public ITakesDynamicAxis
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembers;

    void Init(const TensorShape& sampleLayout, bool isSparse, const std::wstring axisName, float learningRateMultiplier = 0)
    {
        m_isSparse = isSparse;
        MarkValueNonSharable();
        if (isSparse)
            ConvertToSparseMatrix();

        SetDims(sampleLayout, HasMBLayout()); // also called when reloading a file. Then we have an MBLayout, otherwise not yet
        UpdateFunctionValuesSize();           // we must allocate the matrix so that the readers get objects with valid row dimensions (some readers expect that)
        SetLearningRateMultiplier(learningRateMultiplier);
        m_dynamicAxisNodeName = axisName;
    }

protected:
    InputValueBase(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& sampleLayout, bool isSparse, const std::wstring axisName)
        : Base(deviceId, name)
    {
        Init(sampleLayout, isSparse, axisName);
    }
    InputValueBase(DEVICEID_TYPE deviceId, const wstring& name, size_t rows, bool isSparse, const std::wstring axisName)
        : InputValueBase(deviceId, name, TensorShape(rows), isSparse, axisName)
    {
    }
    InputValueBase(DEVICEID_TYPE deviceId, const wstring& name, bool isSparse, const std::wstring axisName)
        : InputValueBase(deviceId, name, TensorShape(), isSparse, axisName)
    {
    }
    InputValueBase(const ScriptableObjects::IConfigRecordPtr configp, bool isSparse)
        : Base(configp->Get(L"deviceId"), L"<placeholder>")
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
        wstring axisName = L"";
        // TODO This currently reads a ComputationNode object from a property, thereby bypassing "normal" input handling.
        // The passing of shapes represents a second graph that is "overlaid" (and previously identical) to the data
        // flow network. This needs to be solved on a more fundamental level.
        // The proposed future change from fseide is as follows:
        // (2) On BS level, dynamicAxis is an optional parameter that takes a DynamicAxis object--the alternative,
        // passing a string, will be removed.
        // (3) The dynamicAxis argument will become an actual m_inputs[] to the InputValue. I.e.InputValues are no
        // longer leaves from the ComputationNetwork viewpoint. But they ARE leaves from the user / BS / NDL view, as
        // the axis is not passed as a regular input.This way, the current special - casing can and will be removed;
        // instead, the MBLayout propagation will happen automagically as part of regular ValidateNetwork().
        if (configp->Exists(L"dynamicAxis"))
        {
            auto axisConfig = configp->Find(L"dynamicAxis");
            if (axisConfig->Is<ComputationNodeBase>())
            {
                ComputationNodeBasePtr axis = configp->Get(L"dynamicAxis");
                axisName = axis->GetName();
            }
            else
            {
                axisName = (const std::wstring&)*axisConfig;
            }
        }

        bool isImage = configp->Get(L"isImage");
        if (!isImage)
            Init(configp->Get(L"shape"), isSparse, axisName);
        else
            Init(ImageDimensions::AsTensorShape(configp->Get(L"imageWidth"), configp->Get(L"imageHeight"), configp->Get(L"imageChannels"), ImageLayoutKindFrom(configp->Get(L"imageLayout"))), isSparse, axisName);
    }

public:
    virtual const std::wstring GetRequestedDynamicAxis() const { return m_dynamicAxisNodeName; }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        size_t rowsDummy = 0; // compat with old file format
        size_t colsDummy = 0;
        fstream << rowsDummy << colsDummy;
        m_sampleLayout.Save(fstream);

        unsigned int nrAxes = 1;
        fstream << nrAxes;
        fstream << m_dynamicAxisNodeName;

        fstream << m_learningRateMultiplier;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);

        size_t rows, colsDummy;
        fstream >> rows >> colsDummy;
        TensorShape sampleLayout;
        sampleLayout.Load(fstream, /*acceptLegacyFormat=*/true);
        // some older files may have inconsistent tensor information
        if (rows != 0 /*old file*/ && rows != sampleLayout.GetNumElements() /*even older file*/)
        {
            fprintf(stderr, "WARNING: %ls InputValue has inconsistent serialized sample layout %s vs. number of rows %d. Resetting sample layout to vector.\n",
                NodeName().c_str(), string(sampleLayout).c_str(), (int)rows);
            sampleLayout = TensorShape(rows);
        }

        if (modelVersion >= CNTK_MODEL_VERSION_8)
        { 
            unsigned int nrAxes;
            fstream >> nrAxes;
            if (nrAxes == 1)
                fstream >> m_dynamicAxisNodeName;
            else if (nrAxes > 1)
                RuntimeError("Input node: This version only supports a single dynamic axis. Please update your bits.");
        }
        else
            m_dynamicAxisNodeName = L""; // Use default

        float learningRateMultiplier = 0;
        if (modelVersion >= CNTK_MODEL_VERSION_10)
            fstream >> learningRateMultiplier;

        Init(sampleLayout, m_isSparse, m_dynamicAxisNodeName, learningRateMultiplier);
    }

    // InputValue must not resize its inputs because that might destroy it. It should already have the correct size.
    virtual void UpdateFunctionMBSize() override
    {
        // don't touch our values
        // But take the opportunity for an additional check. Why not.
        if (Value().GetNumRows() != GetSampleLayout().GetNumElements())
            LogicError("UpdateFunctionMBSize: m_value not matching m_sampleLayout");
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange&) override
    {
        // we have been filled by the Reader
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t /*inputIndex*/, const FrameRange&)
    {
        LogicError("%ls is a leaf node. BackpropTo() should never be called.", NodeName().c_str());
    }

    virtual void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);
        if (printMetadata)
        {
            fstream << "[" << string(GetSampleLayout()) << "]";
        }
    }

private:
    bool m_isSparse = false;
    std::wstring m_dynamicAxisNodeName;
    ComputationNodeBase* m_dynamicAxisNode;

    void ConvertToSparseMatrix()
    {
        m_value->SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, false);
    }
};

// -----------------------------------------------------------------------
// InputValue (/*no input*/)
// an input value (typically fed by a DataReader)
// this covers two types: (regular vs. image)
// TODO: There is still debate whether an InputValue without layout makes sense.
// -----------------------------------------------------------------------

template <class ElemType>
class InputValue : public InputValueBase<ElemType>, public IdentityTransformerNode
{
    typedef InputValueBase<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"InputValue"; }

public:
    InputValue(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name, false, L"")
    {
    }
    InputValue(DEVICEID_TYPE deviceId, const wstring& name, const wstring& dynamicAxisName)
        : Base(deviceId, name, false, dynamicAxisName)
    {
    }
    InputValue(DEVICEID_TYPE deviceId, const wstring& name, size_t rows, const wstring& dynamicAxisName)
        : Base(deviceId, name, rows, false, dynamicAxisName)
    {
    }
    InputValue(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& sampleLayout, const wstring& dynamicAxisName)
        : Base(deviceId, name, sampleLayout, false, dynamicAxisName)
    {
    }
    InputValue(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp, false)
    {
    }
};

template class InputValue<float>;
template class InputValue<double>;

// -----------------------------------------------------------------------
// SparseInputValue (/*no input*/)
// a sparse input value (typically fed by a DataReader)
// this covers two types: (regular vs. image)
// -----------------------------------------------------------------------

template <class ElemType>
class SparseInputValue : public InputValueBase<ElemType>
{
    typedef InputValueBase<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SparseInputValue";
    }

public:
    SparseInputValue(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name, true, L"")
    {
    }
    SparseInputValue(DEVICEID_TYPE deviceId, const wstring& name, const wstring& dynamicAxisName)
        : Base(deviceId, name, true, dynamicAxisName)
    {
    }
    SparseInputValue(DEVICEID_TYPE deviceId, const wstring& name, size_t rows, const wstring& dynamicAxisName)
        : Base(deviceId, name, rows, true, dynamicAxisName)
    {
    }
    SparseInputValue(DEVICEID_TYPE deviceId, const wstring& name, const TensorShape& imageLayout, const wstring& dynamicAxisName)
        : Base(deviceId, name, imageLayout, true, dynamicAxisName)
    {
    }
    SparseInputValue(const ScriptableObjects::IConfigRecordPtr configp)
        : Base(configp, true)
    {
    }
};

template class SparseInputValue<float>;
template class SparseInputValue<double>;

// -----------------------------------------------------------------------
// EnvironmentInput (propertyName) -- read out environment properties
// Such as whether we are currently training or evaluating, which can affect
// behavior, such as seq-2-seq decoding.
// -----------------------------------------------------------------------

template <class ElemType>
class EnvironmentInputNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<0>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"EnvironmentInput"; }

public:
    EnvironmentInputNode(DEVICEID_TYPE deviceId, const wstring& name, const wstring& propertyName = L"") :
        Base(deviceId, name), m_propertyName(propertyName)
    {
    }
    EnvironmentInputNode(const ScriptableObjects::IConfigRecordPtr configp)
        : EnvironmentInputNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"propertyName"))
    {
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_propertyName;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_propertyName;
    }

private:
    ElemType ReadOutVariable() const
    {
        const auto& e = Environment();
        if (m_propertyName == L"isTraining")
            return (ElemType)e.IsTraining();
        else
            InvalidArgument("EnvironmentInput: There is no environment property '%ls'", m_propertyName.c_str());
    }

public:
    // TODO: Noone else overrides this method. So is this the right mechanism?
    //       On the other hand, we are also the only leaf that needs to update itself.
    virtual bool /*ComputationNodeBase::*/ IsOutOfDateWrtInputs() const override { return true; }

    virtual void /*IComputationNode::*/ BeginForwardProp() override
    {
        // We are a leaf, so UpdateFunctionValuesSize() won't be called for us.
        UpdateFunctionValuesSize();
        Base::BeginForwardProp();
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        ElemType val = ReadOutVariable();
        Value().VerifySize(1, 1);
        Value().SetValue(val);
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        LogicError("%ls %ls operation is a leaf node. BackpropTo() should never be called.", NodeName().c_str(), OperationName().c_str());
    }
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ReadOutVariable(); // read out the value once, with the purpose of validating the variableName
        Base::Validate(isFinalValidationPass);
        // this node does not hold mini-batch data
        m_pMBLayout = nullptr;
        // for now, anything this node returns is a scalar
        SetDims(TensorShape(1), false);
    }

private:
    wstring m_propertyName;
};

// -----------------------------------------------------------------------
// LookupTableNode (embedding matrix, bag-of-word representation of the inputs)
// Implements an embedding. The input vector can consist of multiple stacked
// This is a tensor product where the matrix width may be an integer fraction of the features.
// If it is, then the matrix will be replicated.
// This is the same as if the input data were a tensor where the same matrix is applied to each column of the tensor.
// TimesNode can do that.
// -----------------------------------------------------------------------

template <class ElemType>
class LookupTableNode : public ComputationNode<ElemType>, public NumInputs<2>
{
    typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"LookupTable"; }

public:
    DeclareConstructorFromConfigWithNumInputs(LookupTableNode);
    LookupTableNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& t) override
    {
        if (inputIndex == 0) // left derivative (embedding matrix)
        {
            // This is a reduction operation, hence we need to mask out gaps.
            Matrix<ElemType> sliceInput1Value = InputRef(1).MaskedValueFor(t);
            Matrix<ElemType> sliceOutputGrad = MaskedGradientFor(t);

            BackpropToLeft(sliceInput1Value, InputRef(0).GradientAsMatrix(), sliceOutputGrad);
        }
        else if (inputIndex == 1) // right derivative (input)
        {
            Matrix<ElemType> sliceInput1Grad = InputRef(1).GradientFor(t);
            Matrix<ElemType> sliceOutputGrad = GradientFor(t);

            BackpropToRight(InputRef(0).ValueAsMatrix(), sliceInput1Grad, sliceOutputGrad);
        }
    }

    /*TODO: merge with call site*/ void BackpropToLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)
    {
        size_t rows1 = inputFunctionValues.GetNumRows(), cols1 = inputFunctionValues.GetNumCols();
        size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
        int wordsInEachSample = rows1 / inputGradientValues.GetNumCols();

        inputFunctionValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
        gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

        Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, inputFunctionValues, true, inputGradientValues);

        inputFunctionValues.Reshape(rows1, cols1);
        gradientValues.Reshape(rowsp, colsp);
    }

    /*TODO: merge with call site*/ void BackpropToRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues)
    {
        size_t rows1 = inputGradientValues.GetNumRows(), cols1 = inputGradientValues.GetNumCols();
        size_t rowsp = gradientValues.GetNumRows(), colsp = gradientValues.GetNumCols();
        int wordsInEachSample = rows1 / inputFunctionValues.GetNumCols();

        inputGradientValues.Reshape(rows1 / wordsInEachSample, cols1 * wordsInEachSample);
        gradientValues.Reshape(rowsp / wordsInEachSample, colsp * wordsInEachSample);

        Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);

        inputGradientValues.Reshape(rows1, cols1);
        gradientValues.Reshape(rowsp, colsp);
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& t) override
    {
        // input0 is the weight (each column is an embedding of one word), input 1 contains m_nbrLooked words in each column (sample)
        Matrix<ElemType> functionValues =           ValueFor(t);
        const Matrix<ElemType>&  input0 = InputRef(0).ValueAsMatrix();
        Matrix<ElemType>         input1 = InputRef(1).ValueFor(t);

        size_t rows1 = input1.GetNumRows(), cols1 = input1.GetNumCols();
        size_t cols0 = input0.GetNumCols();

        int wordsInEachSample = rows1 / cols0;

        if (cols0 * wordsInEachSample != rows1)
            LogicError("LookupTableNode: rows of input 1 is not a multiple of cols of input 0. This usually happens when the feature dimension is not specified as that in the network definition of look-up-table dimension size.");

        auto input1Reshaped = input1.Reshaped(rows1 / wordsInEachSample, cols1 * wordsInEachSample); // BUGBUG: Won't work for sparse. Also kills BOTH state that we would like to retain.

        auto functionValuesReshaped = functionValues.Reshaped(input0.GetNumRows(), input1Reshaped.GetNumCols());
        functionValuesReshaped.AssignProductOf(input0, false, input1Reshaped, false);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        if (isFinalValidationPass && !HasMBLayout())
            InvalidArgument("%ls %ls operation can only operate on minibatches.", NodeName().c_str(), OperationName().c_str());
        if (isFinalValidationPass && Input(1)->GetSampleMatrixNumRows() % Input(0)->GetAsMatrixNumCols() != 0)
            InvalidArgument("Mismatched dimension. Rows in input1 must be multiples of cols in input0.");

        size_t wordsInEachSample = Input(1)->GetSampleMatrixNumRows() / Input(0)->GetAsMatrixNumCols() /*note: can never be 0*/;

        // TODO: Should this add a tensor dimension?
        SetDims(TensorShape(Input(0)->GetAsMatrixNumRows() * wordsInEachSample), true);
    }

    bool UnitTest()
    {
        try
        {
            size_t nInput = 2;
            size_t nHidden = 3;
            size_t nOutput = 3;

            Input(0)->SetDims1(nInput, nHidden);
            Input(0)->UpdateFunctionValuesSize();
            Input(0)->Value().SetValue(1.0);
            Input(1)->Value().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
            Input(1)->Value().SwitchToMatrixType(DENSE, matrixFormatDense, false);
            Input(1)->SetDims1(nHidden, nOutput);
            Input(1)->UpdateFunctionValuesSize();
            Input(1)->Value().SetValue(0.0);
            Input(1)->Value().SetValue(0, 0, 1.0);
            Input(1)->Value().SetValue(1, 1, 2.0);
            Input(1)->Value().TransferFromDeviceToDevice(CPUDEVICE, m_deviceId, true);
            Input(1)->Value().SwitchToMatrixType(SPARSE, matrixFormatSparseCSC, true);
            SetDims1(nInput, nOutput);
            UpdateFunctionValuesSize();

            ForwardProp(FrameRange(m_pMBLayout));

            // check with expected values
            Value().TransferFromDeviceToDevice(m_deviceId, CPUDEVICE, true);
            if (!ISCLOSE(Value()(0, 0), 1.0, EPSILON) ||
                !ISCLOSE(Value()(0, 1), 2.0, EPSILON) ||
                !ISCLOSE(Value()(1, 1), 2.0, EPSILON))
                throw("LSTMNode forward computation error");

            Value().TransferToDeviceIfNotThere(m_deviceId, true);

            Gradient().Resize(nInput, nOutput);
            Gradient().SetValue(1.0);
            for (size_t i = 0; i < 2; i++)
            {
                Input(i)->Gradient().Resize(Input(i)->Value().GetNumRows(), Input(i)->Value().GetNumCols());
                Input(i)->Gradient().SetValue(0);
            }
            for (size_t i = 0; i < 2; i++)
                BackpropTo(i, FrameRange(m_pMBLayout));

            // check with expected values
            if (!ISCLOSE(Input(1)->Gradient()(0, 0), 2, EPSILON)    // bi
                || !ISCLOSE(Input(1)->Gradient()(0, 1), 2, EPSILON) // Wxi
                || !ISCLOSE(Input(1)->Gradient()(1, 0), 2, EPSILON) // Whi
                || !ISCLOSE(Input(1)->Gradient()(2, 1), 2, EPSILON) // Wci
                )
                throw("LSTMNode gradient error on input gates");

            for (size_t i = 0; i < 2; i++)
                Input(i)->Gradient().TransferToDeviceIfNotThere(m_deviceId, true);
        }
        catch (...)
        {
            fprintf(stderr, "LookupTableNode unit test is not passed!");
            return false;
        }

        fprintf(stderr, "LookupTableNode unit test passed!\n");
        return true;
    }
};

template class LookupTableNode<float>;
template class LookupTableNode<double>;

}}}
