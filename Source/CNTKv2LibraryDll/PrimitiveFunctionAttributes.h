//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "PrimitiveFunction.h"

namespace CNTK
{
    // Names for the reduction operations as used by the CNTK ReduceElementsNode
    /*static*/ const std::wstring PrimitiveFunction::InternalSumReductionOpName = L"Sum";
    /*static*/ const std::wstring PrimitiveFunction::InternalLogSumReductionOpName = L"LogSum";
    /*static*/ const std::wstring PrimitiveFunction::InternalMeanReductionOpName = L"Mean";
    /*static*/ const std::wstring PrimitiveFunction::InternalMaxReductionOpName = L"Max";
    /*static*/ const std::wstring PrimitiveFunction::InternalMinReductionOpName = L"Min";
    /*static*/ const std::wstring PrimitiveFunction::InternalProdReductionOpName = L"Prod";
    /*static*/ const std::wstring PrimitiveFunction::InternalAllReductionOpName = L"All";
    /*static*/ const std::wstring PrimitiveFunction::InternalAnyReductionOpName = L"Any";
    /*static*/ const std::wstring PrimitiveFunction::InternalArgmaxReductionOpName = L"Argmax";
    /*static*/ const std::wstring PrimitiveFunction::InternalArgminReductionOpName = L"Argmin";

    // Names of the various attributes of CNTK primitive Functions
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis = L"axis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxisVec = L"axisVec";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis1 = L"axis1";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis2 = L"axis2";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAllowDuplicates = L"allowDuplicates";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNumSamples = L"numSamples";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDropoutRate = L"dropoutRate";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewShape = L"newShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeginAxis = L"beginAxis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEndAxis = L"endAxis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOutputRank = L"outputRank";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameInferInputRankToMap = L"inferInputRankToMap";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOffset = L"offset";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameStrides = L"strides";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDilation = L"dilation";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSharing = L"sharing";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAutoPadding = L"autoPadding";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameLowerPad = L"lowerPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUpperPad = L"upperPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameCeilOutDim = L"ceilOutDim";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameIncludePad = L"includePad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameTranspose = L"transpose";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOutputShape = L"outputShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples = L"maxTempMemSizeInSamples";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameROIOutputShape = L"roiOutputShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingType = L"poolingType";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingWindowShape = L"poolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSpatial = L"spatial";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNormalizationTimeConstant = L"normalizationTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBlendTimeConstant = L"blendTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEpsilon = L"epsilon";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUseCuDNNEngine = L"useCuDNNEngine";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDisableRegularization = L"disableRegularization";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewDataType = L"newDataType";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewDynamicAxes = L"newDynamicAxes";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewSequenceAxisLengthScalingFactor = L"newSequenceAxisLengthScalingFactor";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewSequenceAxisLengthAdditiveFactor = L"newSequenceAxisLengthAdditiveFactor";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeginIndex = L"beginIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeginIndexVec = L"beginIndexVec";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEndIndex = L"endIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEndIndexVec = L"endIndexVec";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameReductionOpName = L"reductionOpName";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameReductionKeepDimensions = L"reductionKeepDimensions";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBidirectional = L"bidirectional";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNumLayers = L"numLayers";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameHiddenSize = L"hiddenSize";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRecurrentOp = L"recurrentOp";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRngSeed = L"rngSeed";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRngOffset = L"rngOffset";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUnpoolingWindowShape = L"unpoolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSubstitutionPenalty = L"SubstitutionPenalty";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDeletionPenalty = L"DeletionPenalty";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameInsertionPenalty = L"InsertionPenalty";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSquashInputs = L"SquashInputs";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameTokensToIgnore = L"TokensToIgnore";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDelayConstraint = L"DelayConstraint";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBlankTokenId = L"BlankTokenId";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePhonePath = L"PhonePath";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSymListPath = L"SymListPath";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameStateListPath = L"StateListPath";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameTransProbPath = L"TransProbPath";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameLatticeConfigPath = L"LatticeConfigPath";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameHSmoothingWeight = L"HSmoothingWeight";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameFrameDropThresh = L"FrameDropThresh";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDoReferenceAlign = L"DoReferenceAlign";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSeqGammarUsesMBR = L"SeqGammarUsesMBR";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSeqGammarAMF = L"SeqGammarAMF";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSeqGammarLMF = L"SeqGammarLMF";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSeqGammarBMMIFactor = L"SeqGammarBMMIFactor";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSeqGammarWordPen = L"SeqGammarWordPen";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNumClass = L"numClass";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOneHotOutputSparse = L"oneHotOutputSparse";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOneHotAxis = L"onehotAxis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSequenceAxisNamePrefix = L"sequenceAxis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSequenceUnpackPaddingValue = L"sequenceUnpackPaddingValue";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSequenceUnpackSuppressMaskOutput = L"sequenceUnpackSuppressMaskOutput";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRandomDistributionType = L"randomDistributionType";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRandomDistributionArgs = L"randomDistributionArgs";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSpatialScale = L"spatialScale";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSliceStrides = L"sliceStrides";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSliceStridesVec = L"sliceStridesVec";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePaddingHead = L"paddingHead";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePaddingFoot = L"paddingFoot";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePaddingMode = L"paddingMode";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePaddingConstantValue = L"paddingConstantValue";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAlpha = L"alpha";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeta = L"beta";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameGamma = L"gamma";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameKernelShape = L"kernelShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBias = L"bias";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDepthRadius = L"depthRadius";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBlockSize = L"blockSize";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameCustomAttributes = L"customAttributes";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNumItems = L"numItems";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameFillValue = L"fillValue";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUseStatsAcrossChannels = L"useStatsAcrossChannels";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDoVarianceScaling = L"doVarianceScaling";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameGroups = L"groups";
}
