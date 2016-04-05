// ComputationNetworkBuilder -- helper class for constructing ComputationNetworks and ComputationNodes from C++ (internal and external)

// This is used by NDL and the SimpleNetworkBuilder. It will not be used by BrainScript except for New{Standard}Node().

#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "ComputationNetwork.h"
#include "TrainingNodes.h"      // for NCEEvalMode
#include "ConvolutionalNodes.h" // for PoolKind
#include "ScriptableObjects.h"
#include "TensorShape.h"
#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class ComputationNetworkBuilder
{
    typedef shared_ptr<ComputationNode<ElemType>> ComputationNodePtr;
    ComputationNetwork& net;
    ComputationNetworkBuilder();
    ComputationNetworkBuilder(const ComputationNetworkBuilder&);
    void operator=(const ComputationNetworkBuilder&);

public:
    ComputationNetworkBuilder(ComputationNetwork& net)
        : net(net)
    {
    }

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

    // TODO: can these be changed to ComputationNodeBasePtr?
    // TODO: move into a separate header/class, to decouple from this class which would then be only used by old NDL and SimpleNetworkBuilder.
    static ComputationNodePtr NewStandardNode(const std::wstring& nodeType, DEVICEID_TYPE deviceId, const wstring& name);
    static ComputationNodePtr NewNode(const std::wstring& nodeType, DEVICEID_TYPE deviceId, const wstring& name);

    // The following functions create nodes and add them to the net, but don't attach inputs (some don't have inputs).
    // There are special versions for nodes with custom constructors, and a catch-all, CreateComputationNode(), for all others.
    // TODO: Do we really need these? Folks who want to use C++ can instead say net->AddNodeToNet(New<>(...)), which is not that different.
    // TODO: separate into nodes that have inputs and those that duplicate functions with input adding except just not adding inputs. Clear?

    ComputationNodePtr CreateLearnableParameter(const std::wstring& paramName, const size_t rows, const size_t cols);
    ComputationNodePtr CreateLearnableParameter(const std::wstring& paramName, const TensorShape& tensorShape);
    // sparse matrix size is optionally specified
    // ComputationNodePtr CreateSparseLearnableParameter(const std::wstring & paramName, const size_t rows, const size_t cols, const size_t size = 0);
    ComputationNodePtr CreateInputNode(const std::wstring& inputName, const size_t rows);
    ComputationNodePtr CreateSparseInputNode(const std::wstring& inputName, const size_t rows);
    ComputationNodePtr CreateInputNode(const std::wstring& inputName, const TensorShape& sampleLayout);
    ComputationNodePtr CreateSparseInputNode(const std::wstring& inputName, const TensorShape& sampleLayout);
    ComputationNodePtr CreateConvolutionNode(const std::wstring& nodeName, const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                                             const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                             ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples);
    ComputationNodePtr CreateConvolutionNode(const std::wstring& nodeName, const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, 
                                             const size_t horizontalSubsample, const size_t verticalSubsample, 
                                             ImageLayoutKind imageLayoutKind, const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0);
    ComputationNodePtr CreatePoolingNode(const std::wstring& nodeName, PoolKind poolKind, const TensorShape& kernelShape, const TensorShape& strideShape,
                                         const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                         ImageLayoutKind imageLayout);
    ComputationNodePtr CreateMaxPoolingNode(const std::wstring& nodeName, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind);
    ComputationNodePtr CreateAveragePoolingNode(const std::wstring& nodeName, const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind);
    // this is the catch-all for all cases not covered as special cases above
    // Unlike the specialized ones above, this one creates nodes by type given as a string.
    ComputationNodePtr CreateComputationNode(const std::wstring& nodeType, const std::wstring& nodeName);
    // The following functions create nodes and link them to the network and their inputs.
    // TODO: Do we need both this set and the one above that does not add inputs? Can they share more code?
    ComputationNodePtr BatchNormalization(const ComputationNodePtr input, const ComputationNodePtr scale, const ComputationNodePtr bias,
                                          const ComputationNodePtr runMean, const ComputationNodePtr runInvStdDev, bool spatial = false, double normalizationTimeConstant = 0, double blendTimeConstant = 0, double epsilon = 1e-5, bool useCntkEngine = true,
                                          ImageLayoutKind imageLayoutKind = ImageLayoutKind::CHW, const std::wstring nodeName = L"");
    ComputationNodePtr Convolution(const ComputationNodePtr weight,
                                   const ComputationNodePtr inputValues,
                                   const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                                   const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
                                   const bool zeroPadding = false, const size_t maxTempMemSizeInSamples = 0,
                                   const std::wstring nodeName = L"");
    ComputationNodePtr Convolution(const ComputationNodePtr weight,
                                   const ComputationNodePtr inputValues,
                                   const TensorShape& kernelShape, const TensorShape& mapCount, const TensorShape& strideShape,
                                   const std::vector<bool>& sharing, const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                   ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples,
                                   const std::wstring nodeName = L"");
    ComputationNodePtr Pooling(const ComputationNodePtr inputValues, 
                               PoolKind poolKind, const TensorShape& kernelShape, const TensorShape& strideShape,
                               const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                               ImageLayoutKind imageLayout,
                               const std::wstring nodeName = L"");
    ComputationNodePtr MaxPooling(const ComputationNodePtr inputValues,
                                  const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
                                  const std::wstring nodeName = L"");
    ComputationNodePtr AveragePooling(const ComputationNodePtr inputValues,
                                      const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
                                      const std::wstring nodeName = L"");
#ifdef COMING_SOON
    ComputationNodePtr CRF(const ComputationNodePtr label, const ComputationNodePtr postDepScore, const ComputationNodePtr transition_score, const std::wstring nodeName = L"");
#endif
    ComputationNodePtr Abs(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr ClassCrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr input_weight, const ComputationNodePtr cls_log_post_prob, const std::wstring nodeName = L"");
    ComputationNodePtr Cos(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr CosDistance(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr CrossEntropy(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"");
    ComputationNodePtr CrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName = L"");
    ComputationNodePtr DiagTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr Diagonal(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Dropout(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr DummyCriterion(const ComputationNodePtr objectives, const ComputationNodePtr derivatives, const ComputationNodePtr prediction, const std::wstring nodeName = L"");
    ComputationNodePtr ElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr ErrorPrediction(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr Exp(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr FutureValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, size_t timeStep, const std::wstring nodeName = L"");
#ifdef COMING_SOON
    ComputationNodePtr GMMLogLikelihood(const ComputationNodePtr unnormedPrior, const ComputationNodePtr mean, const ComputationNodePtr logStddev, const ComputationNodePtr feature, const std::wstring nodeName = L"");
#endif
    ComputationNodePtr Hardmax(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr InvStdDev(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr KhatriRaoProduct(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr Log(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr LogSoftmax(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Logistic(const ComputationNodePtr a, const ComputationNodePtr b, const ComputationNodePtr c, const std::wstring nodeName = L"");
    ComputationNodePtr Logistic(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr LookupTable(const ComputationNodePtr dictionary, const ComputationNodePtr input, const std::wstring nodeName = L"");
    ComputationNodePtr MatrixL1Reg(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr MatrixL2Reg(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Mean(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Minus(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr Negate(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr NoiseContrastiveEstimation(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr input_weight, const ComputationNodePtr input_bias, const std::wstring nodeName = L"", NCEEvalMode mode = NCEEvalMode::None);
    ComputationNodePtr Pass(const ComputationNodePtr a, const std::wstring& nodeName = L"");
    ComputationNodePtr PastValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, size_t timeStep, const std::wstring nodeName = L"");
    ComputationNodePtr PerDimMeanVarDeNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev, const std::wstring nodeName = L"");
    ComputationNodePtr PerDimMeanVarNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean, const ComputationNodePtr InvStdDev, const std::wstring nodeName = L"");
    ComputationNodePtr Plus(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr Reciprocal(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr RectifiedLinear(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Reshape(const ComputationNodePtr a, const TensorShape& imageLayout, const std::wstring nodeName = L"");
    ComputationNodePtr RowRepeat(const ComputationNodePtr a, const size_t num_repeat, const std::wstring nodeName = L"");
    ComputationNodePtr RowSlice(const ComputationNodePtr a, const size_t start_index, const size_t num_rows, const std::wstring nodeName = L"");
    ComputationNodePtr RowStack(const std::vector<ComputationNodePtr> pinputs, const std::wstring nodeName = L"");
#ifdef COMING_SOON
    ComputationNodePtr SequenceDecoder(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr pairscore, const std::wstring nodeName = L"");
#endif
    ComputationNodePtr SequenceWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr loglikelihood, const std::wstring nodeName = L"");
    ComputationNodePtr Sigmoid(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Softmax(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Sqrt(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr SquareError(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
    ComputationNodePtr Sum(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Tanh(const ComputationNodePtr a, const std::wstring nodeName = L"");
    ComputationNodePtr Times(const ComputationNodePtr a, const ComputationNodePtr b, size_t outputRank = 1, const std::wstring nodeName = L"");
    ComputationNodePtr TransposeDimensions(const ComputationNodePtr matrix, int dim1, int dim2, const std::wstring nodeName = L"");
    ComputationNodePtr TransposeTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName = L"");
#if 1 // legacy
    ComputationNodePtr LegacyReshape(const ComputationNodePtr a, const size_t num_rows, const TensorShape& imageLayout, const std::wstring nodeName = L"");
#endif
};

// create a new from config
shared_ptr<ComputationNodeBase> NewComputationNodeFromConfig(const Microsoft::MSR::ScriptableObjects::IConfigRecordPtr configp);
} } }
