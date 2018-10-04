//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "PrimitiveFunctionAttribute.h"

namespace CNTK
{
    // Names for the reduction operations as used by the CNTK ReduceElementsNode
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalSumReductionOpName = L"Sum";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalLogSumReductionOpName = L"LogSum";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalMeanReductionOpName = L"Mean";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalMaxReductionOpName = L"Max";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalMinReductionOpName = L"Min";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalProdReductionOpName = L"Prod";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalAllReductionOpName = L"All";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalAnyReductionOpName = L"Any";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalArgmaxReductionOpName = L"Argmax";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::InternalArgminReductionOpName = L"Argmin";

    // Names of the various attributes of CNTK primitive Functions
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameAxis = L"axis";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameAxisVec = L"axisVec";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameAxis1 = L"axis1";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameAxis2 = L"axis2";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameAllowDuplicates = L"allowDuplicates";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNumSamples = L"numSamples";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDropoutRate = L"dropoutRate";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNewShape = L"newShape";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBeginAxis = L"beginAxis";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameEndAxis = L"endAxis";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameOutputRank = L"outputRank";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameInferInputRankToMap = L"inferInputRankToMap";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameOffset = L"offset";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameStrides = L"strides";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDilation = L"dilation";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSharing = L"sharing";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameAutoPadding = L"autoPadding";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSequential = L"sequential";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameLowerPad = L"lowerPad";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameUpperPad = L"upperPad";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameCeilOutDim = L"ceilOutDim";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameIncludePad = L"includePad";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameTranspose = L"transpose";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameOutputShape = L"outputShape";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameMaxTempMemSizeInSamples = L"maxTempMemSizeInSamples";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameROIOutputShape = L"roiOutputShape";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNamePoolingType = L"poolingType";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNamePoolingWindowShape = L"poolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSpatial = L"spatial";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNormalizationTimeConstant = L"normalizationTimeConstant";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBlendTimeConstant = L"blendTimeConstant";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameEpsilon = L"epsilon";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameUseCuDNNEngine = L"useCuDNNEngine";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDisableRegularization = L"disableRegularization";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNewDataType = L"newDataType";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNewDynamicAxes = L"newDynamicAxes";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNewSequenceAxisLengthScalingFactor = L"newSequenceAxisLengthScalingFactor";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNewSequenceAxisLengthAdditiveFactor = L"newSequenceAxisLengthAdditiveFactor";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBeginIndex = L"beginIndex";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBeginIndexVec = L"beginIndexVec";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameEndIndex = L"endIndex";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameEndIndexVec = L"endIndexVec";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameReductionOpName = L"reductionOpName";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameReductionKeepDimensions = L"reductionKeepDimensions";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBidirectional = L"bidirectional";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNumLayers = L"numLayers";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameHiddenSize = L"hiddenSize";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameRecurrentOp = L"recurrentOp";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameRngSeed = L"rngSeed";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameRngOffset = L"rngOffset";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameUnpoolingWindowShape = L"unpoolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSubstitutionPenalty = L"SubstitutionPenalty";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDeletionPenalty = L"DeletionPenalty";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameInsertionPenalty = L"InsertionPenalty";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSquashInputs = L"SquashInputs";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameTokensToIgnore = L"TokensToIgnore";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDelayConstraint = L"DelayConstraint";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBlankTokenId = L"BlankTokenId";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNamePhonePath = L"PhonePath";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSymListPath = L"SymListPath";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameStateListPath = L"StateListPath";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameTransProbPath = L"TransProbPath";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameLatticeConfigPath = L"LatticeConfigPath";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameHSmoothingWeight = L"HSmoothingWeight";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameFrameDropThresh = L"FrameDropThresh";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDoReferenceAlign = L"DoReferenceAlign";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSeqGammarUsesMBR = L"SeqGammarUsesMBR";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSeqGammarAMF = L"SeqGammarAMF";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSeqGammarLMF = L"SeqGammarLMF";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSeqGammarBMMIFactor = L"SeqGammarBMMIFactor";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSeqGammarWordPen = L"SeqGammarWordPen";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNumClass = L"numClass";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameOneHotOutputSparse = L"oneHotOutputSparse";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameOutputSparse = L"OutputSparse";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameOneHotAxis = L"onehotAxis";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSequenceAxisNamePrefix = L"sequenceAxis";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSequenceUnpackPaddingValue = L"sequenceUnpackPaddingValue";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSequenceUnpackSuppressMaskOutput = L"sequenceUnpackSuppressMaskOutput";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameRandomDistributionType = L"randomDistributionType";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameRandomDistributionArgs = L"randomDistributionArgs";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSpatialScale = L"spatialScale";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSliceStrides = L"sliceStrides";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameSliceStridesVec = L"sliceStridesVec";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNamePaddingHead = L"paddingHead";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNamePaddingFoot = L"paddingFoot";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNamePaddingMode = L"paddingMode";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNamePaddingConstantValue = L"paddingConstantValue";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameAlpha = L"alpha";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBeta = L"beta";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameGamma = L"gamma";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameKernelShape = L"kernelShape";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBias = L"bias";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDepthRadius = L"depthRadius";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameBlockSize = L"blockSize";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameCustomAttributes = L"customAttributes";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameNumItems = L"numItems";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameFillValue = L"fillValue";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameUseStatsAcrossChannels = L"useStatsAcrossChannels";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameDoVarianceScaling = L"doVarianceScaling";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameGroups = L"groups";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameCustomOp = L"customOp";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameTransposeLeftOperand = L"transA";
    /*static*/ const std::wstring PrimitiveFunctionAttribute::AttributeNameTransposeRightOperand = L"transB";

    /*static*/ const std::vector<std::wstring> PrimitiveFunctionAttribute::s_rngStateAttributes =
                   { PrimitiveFunctionAttribute::AttributeNameRngSeed,
                     PrimitiveFunctionAttribute::AttributeNameRngOffset };
}
