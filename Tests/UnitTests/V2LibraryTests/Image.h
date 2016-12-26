#include "CNTKLibrary.h"

using namespace CNTK;

inline FunctionPtr ConvBNLayer(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, size_t hStride, size_t vStride, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
{
    size_t numInputChannels = input.Shape()[input.Shape().Rank() - 1];

    auto convParams = Parameter({ kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount }, DataType::Float, GlorotUniformInitializer(wScale, -1, 2), device);
    auto convFunction = Convolution(convParams, input, { hStride, vStride, numInputChannels });

    auto biasParams = Parameter({ NDShape::InferredDimension }, (float)bValue, device);
    auto scaleParams = Parameter({ NDShape::InferredDimension }, (float)scValue, device);
    auto runningMean = Constant({ NDShape::InferredDimension }, 0.0f, device);
    auto runningInvStd = Constant({ NDShape::InferredDimension }, 0.0f, device);
    return BatchNormalization(convFunction, scaleParams, biasParams, runningMean, runningInvStd, spatial, (double)bnTimeConst, 0.0, 0.000000001 /* epsilon */);
}

inline FunctionPtr ConvBNReLULayer(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, size_t hStride, size_t vStride, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
{
    auto convBNFunction = ConvBNLayer(input, outFeatureMapCount, kernelWidth, kernelHeight, hStride, vStride, wScale, bValue, scValue, bnTimeConst, spatial, device);
    return ReLU(convBNFunction);
}

inline FunctionPtr ProjLayer(Variable wProj, Variable input, size_t hStride, size_t vStride, double bValue, double scValue, size_t bnTimeConst, const DeviceDescriptor& device)
{
    size_t outFeatureMapCount = wProj.Shape()[0];
    auto b = Parameter({ outFeatureMapCount }, (float)bValue, device);
    auto sc = Parameter({ outFeatureMapCount }, (float)scValue, device);
    auto m = Constant({ outFeatureMapCount }, 0.0f, device);
    auto v = Constant({ outFeatureMapCount }, 0.0f, device);

    size_t numInputChannels = input.Shape()[input.Shape().Rank() - 1];

    auto c = Convolution(wProj, input, { hStride, vStride, numInputChannels }, { true }, { false });
    return BatchNormalization(c, sc, b, m, v, true /*spatial*/, (double)bnTimeConst);
}

inline FunctionPtr ResNetNode2(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
{
    auto c1 = ConvBNReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
    auto c2 = ConvBNLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
    auto p = Plus(c2, input);
    return ReLU(p);
}

inline FunctionPtr ResNetNode2Inc(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, Variable wProj, const DeviceDescriptor& device)
{
    auto c1 = ConvBNReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
    auto c2 = ConvBNLayer(c1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

    auto cProj = ProjLayer(wProj, input, 2, 2, bValue, scValue, bnTimeConst, device);

    auto p = Plus(c2, cProj);
    return ReLU(p);
}

// Standard building block for ResNet with identity shortcut(option A).
inline FunctionPtr ResNetNode2A(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
{
    auto conv1 = ConvBNReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);
    auto conv2 = ConvBNLayer(conv1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

    // Identity shortcut followed by ReLU.
    return ReLU(Plus(conv2, input));
}

// Standard building block for ResNet with padding(option B).
inline FunctionPtr ResNetNode2BInc(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst, bool spatial, const DeviceDescriptor& device)
{
    auto conv1 = ConvBNReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
    auto conv2 = ConvBNLayer(conv1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst, spatial, device);

    // Projection convolution layer.
    auto cProj = ConvBNLayer(input, outFeatureMapCount, 1, 1, 2, 2, wScale, bValue, scValue, bnTimeConst, spatial, device);
    return ReLU(Plus(conv2, cProj));
}
