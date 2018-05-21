//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "InputAndParamNodes.h"
#include "File.h"        // for LoadMatrixFromTextFile()
#include "TensorShape.h" // for SmallVector<>
#include "Globals.h"     // for ShouldForceConstantRandomSeed()

#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// LearnableParameter (/*no input*/)
// represents weight matrices and biases
// TODO: add -Node to the class name
// -----------------------------------------------------------------------

template <class ElemType>
void LearnableParameter<ElemType>::InitShape(const TensorShape& shape)
{
    SetDims(shape, false);
    UpdateFunctionValuesSize(); // this allocates the matrix
    Value().Invalidate();
}

enum DistributionType
{
    Uniform = 0,
    Normal = 1,
    TruncNormal = 2,
};

static pair<DistributionType, double/*stddev or range*/> ParseRandomizationType(const wstring& type, size_t fanOut = 1, size_t fanIn = 1);

// constructor from config
// Parameterization is a little wicked. An older version required to specify the type of initialization
// ("uniform|gaussian|...|fixedValue|fromFile|fromLiteral") and then a parameter with a matching name.
// Now, only the matching parameter is sufficient, making it less verbose.
//  - init="uniform|gaussian|..." (random init, scaled by arg initValueScale)
//  - init="zero"
//  - initValue=scalar --> initialize from this value
//  - initValue=array or nested array --> initialize from this value, infer dimensions  --TODO: not implemented yet
//  - initFromFilePath="..." --> read from a data file. This infers the dimensions from the file.
// deprecated:
//  - init="fixedValue",  value from 'value'            --deprecated in favor of just specifying initValue
//  - init="fromFile",    value from 'initFromFilePath' --deprecated in favor of just specifying 'initFromFilePath'
//  - init="fromLiteral", value from 'initFromLiteral'  --deprecated in favor of initValue=array expression
// Random initialization takes two additional optional parameters: initFilterRank, default 0, and initOutputRank, default 1.
// All dimensions that are not amongst the first 'initOutputRank' are considered inputs.
// This is necessary e.g. for convolution.
// 'initOutputRank' can also be negative to denote output dims on the right, to cater to the needs
// of convolution kernels where the output rank is the right-most axis (initOutputRank=-1).
// The forms that infer the dimensions have different BrainScript names. TODO: need one for fromFile
// TODO: All forms that require specified dimensions but contain zeroes (to be updated by graph)
//       will need to do deferred initialization, or have a way to repeat it.
static TensorShape ToTensorShape(const ScriptableObjects::ConfigValuePtr& val);
template <class ElemType>
LearnableParameter<ElemType>::LearnableParameter(const ScriptableObjects::IConfigRecordPtr configp) :
    LearnableParameter(configp->Get(L"deviceId"), L"<placeholder>", ToTensorShape(configp->Get(L"shape")))
{
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs()); // (we have none; this checks that none are provided)
    // Parameter{dims, other optional parameters: learningRateMultiplier=[1|0|float], init=[uniform|gaussian|], initValueScale=[1|float], initValue=[''|float], initFromFilePath=[''|string]}

    // constant vs. parameter (with optional LR scaling)
    if (configp->Exists(L"learningRateMultiplier"))
        SetLearningRateMultiplier(configp->Get(L"learningRateMultiplier"));
    else if (configp->Exists(L"needsGradient") || configp->Exists(L"needGradient") || configp->Exists(L"computeGradient"))
        InvalidArgument("Deprecated parameter names needsGradient|needGradient|computeGradient are not supported in BrainScript. Use learningRateMultiplier instead.");

    // initialization
    wstring initString = configp->Get(L"init");
    wstring initFromFilePath = configp->Get(L"initFromFilePath");
    let& initValue = configp->Get(L"initValue");   // may be empty string, scalar, or array
    // infer the type of the initial value from what other optional args are given
    if (initString.empty())
    {
        if (!initFromFilePath.empty())                       // 'initFromFilePath' given --> initialize from file
            initString = L"fromFile"; // (note: this is only used internally; external use is deprecated)
        else if (!initValue.Is<ScriptableObjects::String>()) // 'initValue' given (not an empty string) --> initialize from value
        {
            if (initValue.Is<ScriptableObjects::Double>())
                initString = L"fromValue"; // (note: this is only used internally)
            else if (initValue.Is<ScriptableObjects::ConfigArray>())
                initString = L"fromValueArray"; // (note: this is only used internally)
            else
                InvalidArgument("'initValue' must be numerical");
        }
        else if (!initValue.AsRef<ScriptableObjects::String>().empty()) // it's a string: must be empty
            InvalidArgument("LearnableParameter: 'initValue' must be an empty string or not a string.");
        else  // no pertinent optional arguments given: default to 'uniform'
            initString = L"uniform"; // default is uniform
    }
    // deferred variants
    // Deferred means that this kind of initialization is allowed when some dimensions are unspecified, and thus happens during Validate().
    if (ParseRandomizationType(initString).second != 0) // random init
    {
        m_initString = initString;
        // TODO: add more randomization types, and use a more meaningful scaling
        // Keras uses "normal" instead of "gaussian". We can use that here too to denote the one with sane scaling, and deprecate "gaussian" with a warning.
        static unsigned long randomSeed = 1;
        int forcedRandomSeed = configp->Get(L"randomSeed"); // forcing a specific random seed is useful for testing to get repeatable initialization independent of evaluation order
        m_randomSeed = forcedRandomSeed < 0 ? randomSeed++ : (unsigned long)forcedRandomSeed;
        m_initValueScale = (ElemType)(float)configp->Get(L"initValueScale");
        m_initFilterRank = configp->Get(L"initFilterRank"); 
        m_initOutputRank = configp->Get(L"initOutputRank");
        m_initOnCPUOnly  = configp->Get(L"initOnCPUOnly");
    }
    else if (initString == L"zero")
    {
        m_initString = L"fromValue";
        m_initValue = (ElemType)0;
    }
    else if (initString == L"fromValue") // from 'initValue'
    {
        m_initString = initString;
        m_initValue = (ElemType)(float)initValue;
    }
    else if (initString == L"bilinear")
    {
        m_initString = initString;
    }
    // non-deferred variants
    // For the dimensions are always known at this point, so we don't need/want to have to save all those parameters.
    else if (initString == L"fromValueArray") // from 'initValue' which has array form
        InvalidArgument("'initValue' for arrays not yet implemented"); // array not yet implemented
    else if (initString == L"fromFile") // load from 'iniFromFilePath'
    {
        if (initFromFilePath.empty())
            RuntimeError("initFromFilePath parameter must be provided when using \"fromFile\" initialization method");
        InitFromFile(initFromFilePath);
        m_initString.clear();
    }
    // legacy
    else if (initString == L"fixedValue") // deprecated. Use initValue=... instead
    {
        m_initString = L"fromValue";
        m_initValue = (ElemType)(float)configp->Get(L"value");
    }
    else if (initString == L"fromLiteral") // deprecated. Use initValue=array instead
    {
        wstring initFromLiteral = configp->Get(L"initFromLiteral");
        if (initFromLiteral.empty())
            RuntimeError("initFromLiteral parameter must be provided when using \"fromLiteral\" initialization method");
        size_t numRows, numCols;
        auto array = File::LoadMatrixFromStringLiteral<ElemType>(msra::strfun::utf8(initFromLiteral), numRows, numCols);
        InitFromArray(array, numRows, numCols);
        m_initString.clear();
    }
    else
        RuntimeError("init must be one of the values of [ uniform | gaussian | fixedValue | fromFile | bilinear ]");

    // initialize
    // This will be repeated if the matrix gets resized due to dimension inference.
    LazyInitParameters();

    auto traceLevelParam = configp->Find(L"traceLevel");
    if (traceLevelParam && (int)*traceLevelParam > 0 && !m_initString.empty())
        fprintf(stderr, "%ls: Initializating Parameter[%s] as %ls later when dimensions are fully known.\n", NodeDescription().c_str(), string(GetSampleLayout()).c_str(), m_initString.c_str());
}

// helper to cast a shape possibly given as a single size_t to a TensorShape object
// This is specifically for use by BrainScript, which, for simplicity, is allowed to pass
// a (size_t)1 when type-casting a scalar constant to a LearnableParameter.
static TensorShape ToTensorShape(const ScriptableObjects::ConfigValuePtr& val)
{
    if (val.Is<TensorShape>())
        return val.AsRef<TensorShape>();
    else
        return TensorShape((size_t)val);
}

// variant of above from NDL. Must be called right after plain constructor.
// This overwrites any pending deferred initialization with a new one.
// Initialization is done immediately if all dimensions are already known, otherwise kept pending.
template <class ElemType>
void LearnableParameter<ElemType>::PostInitParameters(const wstring& initString, // "uniform"|"gaussian"|"fixedValue"
                                                      ElemType initValue,        //  scale   | scale    | value
                                                      unsigned long randomSeed /*= 0*/,
                                                      bool initOnCPUOnly /*= false*/)
{
    if (ParseRandomizationType(initString).second != 0) // random init
    {
        m_initString = initString;
        m_randomSeed = randomSeed;
        m_initValueScale = initValue;
        m_initFilterRank = 0; // default. NDL (deprecated) cannot specify a different value.  
        m_initOutputRank = 1; // default. NDL (deprecated) cannot specify a different value.
        m_initOnCPUOnly = initOnCPUOnly;
    }
    else if (initString == L"fixedValue") // from constant value
    {
        m_initString = L"fromValue";
        m_initValue = initValue;
    }
    else
        LogicError("PostInitParameters: invalid init string '%ls'", m_initString.c_str());

    // initialize
    // This will be repeated if the matrix gets resized due to dimension inference.
    LazyInitParameters();

    if (!m_initString.empty())
        fprintf(stderr, "%ls: Initializating Parameter[%s] as %ls later when dimensions are fully known.\n", NodeDescription().c_str(), string(GetSampleLayout()).c_str(), m_initString.c_str());
}

// understood options:
//  uniformBS:     1/20
//  uniform:       1.0
//  gaussian:      sqrt(0.04 / fanin)
//  normal:        1.0
//  xavier:        sqrt(3 / fanin)
//  glorotNormal:  sqrt(2 / (fanin+fanout))
//  glorotUniform: sqrt(6 / (fanin+fanout))
//  heNormal:      sqrt(2 / fanin)
//  heUniform:     sqrt(6 / fanin)
//  TruncNormal: 1.0    
// returns (Normal,0) for unrecognized string
static pair<DistributionType, double/*stddev or range*/> ParseRandomizationType(const wstring& type, size_t fanOut /* = 1*/, size_t fanIn /*= 1*/)
{
    if      (type == UniformBSInitializerTypeName)     return make_pair(Uniform,     0.05f);
    else if (type == UniformInitializerTypeName)       return make_pair(Uniform,     1.0f);
    else if (type == GaussianInitializerTypeName)      return make_pair(Normal,      0.2 / sqrt(fanIn));
    else if (type == NormalInitializerTypeName)        return make_pair(Normal,      1.0f);
    else if (type == XavierInitializerTypeName)        return make_pair(Uniform,     sqrt(3.0 / fanIn));
    else if (type == GlorotUniformInitializerTypeName) return make_pair(Uniform,     sqrt(6.0 / (fanIn + fanOut)));
    else if (type == GlorotNormalInitializerTypeName)  return make_pair(Normal,      sqrt(2.0 / (fanIn + fanOut)));
    else if (type == HeUniformInitializerTypeName)     return make_pair(Uniform,     sqrt(6.0 / fanIn));
    else if (type == HeNormalInitializerTypeName)      return make_pair(Normal,      sqrt(2.0 / fanIn));
    else if (type == TruncNormalInitializerTypeName)   return make_pair(TruncNormal, 1.0);
    else                                               return make_pair(Normal,      0.0);
}

// initialize with random numbers
// if 'initOnCPUOnly' then always init on CPU, making initialization consistent across both (for testing)
template <class ElemType>
std::tuple<size_t, size_t, ElemType> LearnableParameter<ElemType>::InitRandom(Matrix<ElemType>& valueMatrix,
                                                                              const TensorShape& sampleShape,
                                                                              const wstring& type,
                                                                              const unsigned long randomSeed,
                                                                              const ElemType initValueScale,
                                                                              const size_t initFilterRank,
                                                                              const int initOutputRank,
                                                                              const bool initOnCPUOnly,
                                                                              DEVICEID_TYPE deviceId)
{
    let& sampleLayout = sampleShape;
    let numElements = sampleLayout.GetNumElements();
    if (numElements == 0)
        return std::make_tuple((size_t)0, (size_t)0, (ElemType)0.0f);

    if (initFilterRank + abs(initOutputRank) > sampleLayout.GetRank())
        InvalidArgument("InitRandom: initFilterRank=%d and initOutputRank=%d exceeds sampleLayout rank %d", (int)initFilterRank, initOutputRank, (int)sampleLayout.GetRank());
    // determine fan-in and fan-out
    // In the most generic case of convolution, sampleLayout should be in the form of [f1 x f2 x ... x fr x c1 x ... x cm x k1 x ... x kn],
    // where r is the filterRank, m is the input rank, and n is the output rank. In the above example, we should have initOutputRank = -n
    // If initOutputRank = n, the layout should be [[f1 x f2 x ... x fr x k1 x ... x kn x c1 x ... x cm], note filter dimensions stay in the front of the layout 
    // in the case of dense layers, initFilterRank = r = 0. 
    size_t filterSize = 1; 
    for (size_t k = 0; k < initFilterRank; k++)
        filterSize *= sampleLayout[k];

    let inDimsBegin = initFilterRank + ((initOutputRank >= 0) ? (size_t)initOutputRank : 0);
    let inDimsEnd = (initOutputRank >= 0) ? sampleLayout.GetRank() : (size_t)((int)sampleLayout.GetRank() + initOutputRank);
    size_t fanIn = filterSize; 
    for (size_t k = inDimsBegin; k < inDimsEnd; k++)
        fanIn *= sampleLayout[k];
    size_t fanOut = numElements * filterSize / fanIn;

    let opts = ParseRandomizationType(type, fanOut, fanIn);
    ElemType range = (ElemType)opts.second;
    if (range == 0)
        LogicError("InitRandom: Invalid initialization type '%ls'", type.c_str());

    range *= initValueScale;

    // the random seed offset is set via the "randomSeedOffset" parameter in config
    if (initOnCPUOnly)
        valueMatrix.TransferToDeviceIfNotThere(CPUDEVICE, true);
    if (opts.first == DistributionType::Uniform)
        valueMatrix.SetUniformRandomValue(-range, range, randomSeed);
    else if (opts.first == DistributionType::Normal)
        valueMatrix.SetGaussianRandomValue(0, range, randomSeed);
    else if (opts.first == DistributionType::TruncNormal)
        valueMatrix.SetTruncatedNormalRandomValue(0, range, randomSeed);
    else
        LogicError("InitRandom: Unknown distribution type '%d'", opts.first);
    if (initOnCPUOnly)
        valueMatrix.TransferToDeviceIfNotThere(deviceId, true);

    return std::make_tuple(fanOut, fanIn, range);
}

// Initialize with bilinear interpolation coefficients (useful for deconvolution layer).
template <class ElemType>
void LearnableParameter<ElemType>::InitBilinear(Matrix<ElemType>& valueMatrix, const TensorShape& sampleShape, size_t kernelWidth, size_t kernelHeight, DEVICEID_TYPE deviceId)
{
    if (kernelHeight != kernelWidth)
        LogicError("Filter for bilinear interpolation must be square.");

    // Transfer to CPU as GPU initialization is still not supported.
    valueMatrix.TransferToDeviceIfNotThere(CPUDEVICE, true);

    const SmallVector<size_t>& dims = sampleShape.GetDims();
    assert(dims.size() >= 4);
    assert(dims[0] == kernelWidth);
    assert(dims[1] == kernelHeight);
    const size_t kernelCount = dims[2];
    const size_t channels = dims[3];
    if (kernelCount != channels)
        LogicError("Number of input and output channels of filter for bilinear interpolation must be equal.");

    ElemType* data = valueMatrix.Data();
    const size_t factor = (kernelWidth + 1) / 2;
    const float center = (kernelWidth - 1) / 2.0f;
    int count = 0;
    // Filter dimensions are [W x H x C x K] or ARRAY[1..K] OF ARRAY[1..C] OF ARRAY[1..H] OF ARRAY[1..W], where:
    // W = width, H = height, C = input channels, K = output channels.
    // In deconvolution, output channel should be upsampled version of corresponding input channel.
    // 2D filter for bilinear interpolation where height=width=3 contains the following values:
    // |0.25, 0.50, 0.25|
    // |0.50, 1.00, 0.50|
    // |0.25, 0.50, 0.25|
    // So, output kernel with dimensions [3 x 3 x C] will contain all zeros except for the channel which we want to
    // upsample. For that channel it will contain values above.
    for (size_t kernel = 0; kernel < kernelCount; ++kernel)
    {
        for (size_t channel = 0; channel < channels; ++channel)
        {
            for (size_t h = 0; h < kernelHeight; ++h)
            {
                for (size_t w = 0; w < kernelWidth; ++w)
                {
                    float val = 0;
                    if (kernel == channel)
                    {
                        val = (1 - fabs(w - center) / factor) * (1 - fabs(h - center) / factor);
                    }
                    data[count++] = val;
                }
            }
        }
    }

    valueMatrix.TransferToDeviceIfNotThere(deviceId, true);
}

// Initialize with bilinear interpolation coefficients (useful for deconvolution layer).
template<>
void LearnableParameter<char>::InitBilinear(Matrix<char>& valueMatrix, const TensorShape& sampleShape, size_t kernelWidth, size_t kernelHeight, DEVICEID_TYPE deviceId)
{
    RuntimeError("Unsupported template argument(char) in InitBilinear");
}

template <>
std::tuple<size_t, size_t, char> LearnableParameter<char>::InitRandom(Matrix<char>& valueMatrix,
    const TensorShape& sampleShape,
    const wstring& type,
    const unsigned long randomSeed,
    const char initValueScale,
    const size_t initFilterRank,
    const int initOutputRank,
    const bool initOnCPUOnly,
    DEVICEID_TYPE deviceId)
{
    RuntimeError("Unsupported template argument(char) in InitRandom");
}

// initialize by reading a matrix from a text file
template <class ElemType>
void LearnableParameter<ElemType>::InitFromFile(const wstring& initFromFilePath)
{
    size_t numRows, numCols;
    auto array = File::LoadMatrixFromTextFile<ElemType>(initFromFilePath, numRows, numCols);
    InitFromArray(array, numRows, numCols);
}

// initialize by reading a matrix from a text file
template <class ElemType>
void LearnableParameter<ElemType>::InitFromArray(const vector<ElemType>& array, size_t numRows, size_t numCols)
{
    // infer tensor dimensions from input file if not set
    // Note: The mapping of dimensions of the input matrix to tensor dimensions are somewhat confusing.
    //       The file contains a 2D matrix (one row per text line) that is saved into our column-major representation.
    //       That representation is then reshaped into a column-major tensor.
    if (GetSampleLayout().GetNumElements() == 0)    // at least one dimension is 0
    {
        auto dims = GetSampleLayout().GetDims();
        // infer rank
        if (dims.size() == 0)
            dims.push_back(0);
        if (dims.size() == 1 && numCols != 1)
            dims.push_back(0);
        // infer #rows
        if (dims[0] == 0)           // infer row dimension as input matrix row dimension
            dims[0] = numRows;      // (if already set, then mismatch will be caught in VerifyDataSize() below)
        // infer #cols: product of all dimensions but the first must match matrix #cols; if there is a single 0 position, we infer it
        size_t zeroDim = 0;         // 0 means not found
        size_t prod = 1;
        for (size_t k = 1; k < dims.size(); k++)
        {
            auto dim = dims[k];
            if (dim != 0)
                prod *= dim;
            else if (zeroDim == 0)
                zeroDim = k;
            else
                InvalidArgument("%ls %ls operation's specified shape [%s] cannot be inferred: Too many unknown dimensions.", NodeName().c_str(), OperationName().c_str(), string(GetSampleLayout()).c_str());
        }
        if (zeroDim != 0)   // we found a zero
        {
            dims[zeroDim] = numCols / prod;
            if (prod * dims[zeroDim] != numCols)
                InvalidArgument("%ls %ls operation's specified shape [%s] cannot be inferred: Tensor shape cannot hold a [%d x %d] matrix.", NodeName().c_str(), OperationName().c_str(), string(GetSampleLayout()).c_str(), (int)numRows, (int)numCols);
        }
        SetDims(TensorShape(dims), false);
    }

    // BUGBUG: We should allow to read an arbitrary tensor from a single-column file.
    //         Currently, this would cause a matrix/tensor dimension mismatch. --TODO: Is this comment up-to-date?
    Value().SetValue(numRows, numCols, m_deviceId, const_cast<ElemType*>(array.data()), matrixFlagNormal);
    // TODO: Get rid of that const_cast, as soon as after Ryan's Matrix-lib refactoring separated out SetValue() from external vs. from deep copy
    VerifyDataSize(Value());      // sanity check
}

// TODO: Move this error check there, since this is called only from one place.
template <class ElemType>
void LearnableParameter<ElemType>::ReviseFromFile(const wstring& reviseFromFilePath)
{
    try
    {
        InitFromFile(reviseFromFilePath);
    }
    catch (const exception & e)
    {
        RuntimeError("ReviseFromFile: Failed to reload %ls %ls operation from file %ls: %s", NodeName().c_str(), OperationName().c_str(), reviseFromFilePath.c_str(), e.what());
    }
}

template <class ElemType>
void LearnableParameter<ElemType>::Save(File& fstream) const /*override*/
{
    if (!m_initString.empty())
        LogicError("LearnableParameter: Cannot Save() before deferred initialization has completed.");
    Base::Save(fstream);
    fstream << m_learningRateMultiplier;
    m_sampleLayout.Save(fstream);
    fstream << Value();
}

template <class ElemType>
void LearnableParameter<ElemType>::Load(File& fstream, size_t modelVersion) /*override*/
{
    Base::Load(fstream, modelVersion);

    TensorShape sampleLayout;

    if (modelVersion >= CNTK_MODEL_VERSION_3)
    {
        fstream >> m_learningRateMultiplier;
        sampleLayout.Load(fstream);
    }
    else // legacy format(s)
    {
        bool parameterUpdateRequired;
        fstream >> parameterUpdateRequired;
        SetLearningRateMultiplier((float)parameterUpdateRequired);

        size_t rows, cols;
        fstream >> rows >> cols;
        if (rows != 0) // legacy file format
            sampleLayout = TensorShape(rows, cols);
        else
        {
            sampleLayout.Load(fstream, /*acceptLegacyFormat=*/true);
            if (cols > 1) // in some legacy format, last tensor dimension was split off as an explicit column dimension
                sampleLayout.AppendInPlace(sampleLayout.GetRank(), cols);
        }
    }

    LoadValue(fstream);
    SetDims(sampleLayout, false); // note: call this after LoadValue() since LoadValue() overwrites m_sampleLayout
    VerifyDataSize(Value());      // sanity check

    m_initString.clear(); // deferred initialization not possible after loading
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::CopyTo(ComputationNodeBasePtr nodeP, const wstring& newName, const CopyNodeFlags flags) const /*override*/
{
    Base::CopyTo(nodeP, newName, flags);
    if (flags & CopyNodeFlags::copyNodeValue)
    {
        auto node = dynamic_pointer_cast<LearnableParameter<ElemType>>(nodeP);
        node->m_initString     = m_initString;
        node->m_randomSeed     = m_randomSeed;
        node->m_initValueScale = m_initValueScale;
        node->m_initFilterRank = m_initFilterRank; 
        node->m_initOutputRank = m_initOutputRank;
        node->m_initOnCPUOnly  = m_initOnCPUOnly;
        node->m_initValue      = m_initValue;
    }
}

// computation functions don't do anything for parameter nodes
template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::UpdateFunctionMBSize() /*override*/
{
    if (!m_initString.empty())
        LogicError("LearnableParameter: Deferred initialization has not been completed until first call to UpdateFunctionMBSize().");
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::ForwardProp(const FrameRange&) /*override*/
{
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::BackpropTo(const size_t /*inputIndex*/, const FrameRange&) /*override*/
{
    LogicError("%ls %ls operation is a leaf node. BackpropTo() should never be called.", NodeName().c_str(), OperationName().c_str());
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::Validate(bool isFinalValidationPass) /*override*/
{
    //fprintf(stderr, "Validate %ls: called in init state '%ls' with dims [%s]\n", NodeDescription().c_str(), m_initString.c_str(), string(GetSampleLayout()).c_str());
    Base::Validate(isFinalValidationPass);
    m_pMBLayout = nullptr; // this node does not hold mini-batch data

    // lazy init if we got a dimension now
#if 0 // fake old buggy behavior before deferred initialization
    if (isFinalValidationPass && !m_initString.empty() && (m_initString != L"fromValue" || m_initValue != 0))
    {
        fprintf(stderr, "Validate: deferred '%ls' initialization patched to fromValue 0 for back compat\n", m_initString.c_str());
        m_initString = L"fromValue";
        m_initValue = 0;
    }
#endif
#if 0
    // We call this here and in Validate(true), since we don't know which gets called first.
    // TODO: Actually this should never be needed, because each time dimensions change, we init.
    //       So if we get here without fully-known dimensions, this call won't do anything either.
    if (isFinalValidationPass)
        LazyInitParameters();
#endif
}

// deferred initialization
// We support a feature that some dimensions can be specified as 0, and get inferred.
// This is only possible for initialization methods that do not come with their own dimensions
// (such as initialization from an array literal).
// When initialization succeeded (all dimensions known), the pending initialization is cleared.
// This is called from constructor and InferInputDimsFrom().
// BUGBUG: We cannot really enforce the calling sequence. Save() verifies that this has been cleared.
//         Note that this may be called AFTER Validate(true) (still during validation, but after final validation of this node).
template <class ElemType>
void LearnableParameter<ElemType>::LazyInitParameters()
{
    // if no lazy init pending then we are done
    if (m_initString.empty())
        return;
    // if not all dimensions are known yet, we cannot proceed: keep it pending
    if (GetSampleLayout().GetNumElements() == 0)
        return;
    // OK, proceed
    if (m_initString == L"fromValue")
    {
        if (GetEnvironmentPtr() && Environment().traceLevel > 0) // note: this will not log before node is part of network
            fprintf(stderr, "%ls: Initializing Parameter[%s] <- %f.\n", NodeDescription().c_str(), string(GetSampleLayout()).c_str(), (float)m_initValue);
        Value().SetValue(m_initValue);
    }
    else if (ParseRandomizationType(m_initString).second != 0)
    {
        let randomSeed = Globals::ShouldForceConstantRandomSeed() ? 1UL : m_randomSeed; // debugging feature to enforce identical results across NDL, BrainScript, and V2 API/Python
        InitRandom(m_initString, randomSeed, m_initValueScale, m_initFilterRank, m_initOutputRank, m_initOnCPUOnly);
    }
    else if (m_initString == L"bilinear")
    {
        const TensorShape shape = GetSampleLayout();
        const size_t kernelWidth = shape[0];
        const size_t kernelHeight = shape[1];
        InitBilinear(kernelWidth, kernelHeight);
    }
    else
        LogicError("LearnableParameter: Invalid value of m_initString '%ls' for deferred initialization for %ls.", m_initString.c_str(), NodeDescription().c_str());
    // and remember that we are done
    m_initString.clear();
}

// called from ComputationNode::ValidateInferInputDimsFrom()
// In case of an error, this function just backs out without updating.
// The caller must verify the dimensions.
// This is a bit weird since it is called after this node has been Validated once.
template <class ElemType>
void LearnableParameter<ElemType>::InferInputDimsFrom(const TensorShape& otherShape)
{
//fprintf(stderr, "InferInputDimsFrom %ls: called in init state '%ls' with dims [%s], offered new dims [%s]\n", NodeDescription().c_str(), m_initString.c_str(), string(GetSampleLayout()).c_str(), string(otherShape).c_str());
    const auto& thisShape = GetSampleLayout();

    // see where we stand with our shape
    bool hasMissingDims = thisShape.GetNumElements() == 0;
    if (!hasMissingDims) // all there--nothing to infer
        return;
    
    // infer at least one dimension
    if (otherShape.GetNumElements() == 0)
        return; // LogicError("ValidateInferInputDimsFrom: Inferred dimensions must not be empty.");

    if (m_initString.empty())
        LogicError("InferInputDimsFrom: Attempted to infer dimensions, with initialization completed or no deferred initialization pending.");

    // if no dimensions have been set at all, copy otherShape
    // Don't verify dimensions in this case, because the node may have explicitly been defined as a vector of 0 elements.
    bool hasAnyDim = false;
    for (auto dim : thisShape.GetDims())
        hasAnyDim |= dim != 0;
    if (!hasAnyDim)          // just use it straight
        InitShape(otherShape);
    else if (hasMissingDims) // we got a pre-existing shape: If it has zeroes, we fill them in from otherShape
    {
        if (thisShape.GetRank() != 0 && thisShape.GetRank() > otherShape.GetRank())
            return; // LogicError("ValidateInferInputDimsFrom: Inferred dimensions cannot decrease rank.");
        SmallVector<size_t> newDims = thisShape.GetDims();
        newDims.resize(otherShape.GetRank(), 0);
        for (size_t i = 0; i < newDims.size(); i++)
            if (newDims[i] == 0)
                newDims[i] = otherShape[i];
        InitShape(TensorShape(newDims));
    }
    fprintf(stderr, "%ls operation: Tensor shape was inferred as [%s].\n", NodeDescription().c_str(), string(GetSampleLayout()).c_str());

    // initialize the values
    // We call this here and in Validate(true), since we don't know which gets called first.
    // Note: It seems that this is not necessary, and that Validate(true) is only called after inference.
#if 0 // fake old buggy behavior before deferred initialization
    if (m_initString != L"fromValue" || m_initValue != 0)
    {
        fprintf(stderr, "InferInputDimsFrom: deferred '%ls' initialization patched to fromValue 0 for back compat\n", m_initString.c_str());
        m_initString = L"fromValue";
        m_initValue = 0;
    }
#endif
    LazyInitParameters();
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const /*override*/
{
    if (printMetadata)
    {
        Base::DumpNodeInfo(printValues, printMetadata, fstream);

        char str[4096];
        sprintf(str, "[%lu,%lu]  ", (unsigned long)GetAsMatrixNumRows(), (unsigned long)GetAsMatrixNumCols());
        fstream << string(str);
        sprintf(str, "learningRateMultiplier=%f  NeedsGradient=%s", m_learningRateMultiplier, m_learningRateMultiplier>0 ? "true" : "false"); // TODO: update NDL to accept a better matching name as well
        fstream << string(str);
    }

    PrintNodeValuesToFile(printValues, printMetadata, fstream);
}

template <class ElemType>
/*virtual*/ void LearnableParameter<ElemType>::FreezeParameters() /*override*/ // from IFreezable
{
    SetLearningRateMultiplier(0);
}

template class LearnableParameter<float>;
template class LearnableParameter<double>;
template class LearnableParameter<half>;

}}}
