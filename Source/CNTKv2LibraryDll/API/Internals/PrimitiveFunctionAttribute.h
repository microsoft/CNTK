//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <string>
#include <vector>

#ifndef CNTK_API
#ifdef _WIN32
#ifdef CNTKV2LIBRARYDLL
#define CNTK_API __declspec(dllexport)
#else // !defined(CNTKV2LIBRARYDLL)
#define CNTK_API __declspec(dllimport)
#endif // ifdef(CNTKV2LIBRARYDLL)
#else // no DLLs on Linux
#define CNTK_API
#endif // ifdef(_WIN32)
#endif // ifndef(CNTK_API)

namespace CNTK
{
    class PrimitiveFunctionAttribute {
    public:
        CNTK_API static const std::wstring InternalSumReductionOpName;
        CNTK_API static const std::wstring InternalLogSumReductionOpName;
        CNTK_API static const std::wstring InternalMeanReductionOpName;
        CNTK_API static const std::wstring InternalMaxReductionOpName;
        CNTK_API static const std::wstring InternalMinReductionOpName;
        CNTK_API static const std::wstring InternalProdReductionOpName;
        CNTK_API static const std::wstring InternalAllReductionOpName;
        CNTK_API static const std::wstring InternalAnyReductionOpName;
        CNTK_API static const std::wstring InternalArgmaxReductionOpName;
        CNTK_API static const std::wstring InternalArgminReductionOpName;

        CNTK_API static const std::wstring AttributeNameAxis;
        CNTK_API static const std::wstring AttributeNameAxisVec;
        CNTK_API static const std::wstring AttributeNameAxis1;
        CNTK_API static const std::wstring AttributeNameAxis2;
        CNTK_API static const std::wstring AttributeNameAllowDuplicates;
        CNTK_API static const std::wstring AttributeNameNumSamples;
        CNTK_API static const std::wstring AttributeNameDropoutRate;
        CNTK_API static const std::wstring AttributeNameNewShape;
        CNTK_API static const std::wstring AttributeNameBeginAxis;
        CNTK_API static const std::wstring AttributeNameEndAxis;
        CNTK_API static const std::wstring AttributeNameOutputRank;
        CNTK_API static const std::wstring AttributeNameInferInputRankToMap;
        CNTK_API static const std::wstring AttributeNameOffset;
        CNTK_API static const std::wstring AttributeNameStrides;
        CNTK_API static const std::wstring AttributeNameDilation;
        CNTK_API static const std::wstring AttributeNameSharing;
        CNTK_API static const std::wstring AttributeNameAutoPadding;
        CNTK_API static const std::wstring AttributeNameSequential;
        CNTK_API static const std::wstring AttributeNameLowerPad;
        CNTK_API static const std::wstring AttributeNameUpperPad;
        CNTK_API static const std::wstring AttributeNameCeilOutDim;
        CNTK_API static const std::wstring AttributeNameIncludePad;
        CNTK_API static const std::wstring AttributeNameTranspose;
        CNTK_API static const std::wstring AttributeNameOutputShape;
        CNTK_API static const std::wstring AttributeNameMaxTempMemSizeInSamples;
        CNTK_API static const std::wstring AttributeNameROIOutputShape;
        CNTK_API static const std::wstring AttributeNamePoolingType;
        CNTK_API static const std::wstring AttributeNamePoolingWindowShape;
        CNTK_API static const std::wstring AttributeNameSpatial;
        CNTK_API static const std::wstring AttributeNameNormalizationTimeConstant;
        CNTK_API static const std::wstring AttributeNameBlendTimeConstant;
        CNTK_API static const std::wstring AttributeNameEpsilon;
        CNTK_API static const std::wstring AttributeNameUseCuDNNEngine;
        CNTK_API static const std::wstring AttributeNameDisableRegularization;
        CNTK_API static const std::wstring AttributeNameNewDataType;
        CNTK_API static const std::wstring AttributeNameNewDynamicAxes;
        CNTK_API static const std::wstring AttributeNameNewSequenceAxisLengthScalingFactor;
        CNTK_API static const std::wstring AttributeNameNewSequenceAxisLengthAdditiveFactor;
        CNTK_API static const std::wstring AttributeNameBeginIndex;
        CNTK_API static const std::wstring AttributeNameBeginIndexVec;
        CNTK_API static const std::wstring AttributeNameEndIndex;
        CNTK_API static const std::wstring AttributeNameEndIndexVec;
        CNTK_API static const std::wstring AttributeNameReductionOpName;
        CNTK_API static const std::wstring AttributeNameReductionKeepDimensions;
        CNTK_API static const std::wstring AttributeNameRngSeed;
        CNTK_API static const std::wstring AttributeNameRngOffset;
        CNTK_API static const std::wstring AttributeNameBidirectional;
        CNTK_API static const std::wstring AttributeNameNumLayers;
        CNTK_API static const std::wstring AttributeNameHiddenSize;
        CNTK_API static const std::wstring AttributeNameRecurrentOp;
        CNTK_API static const std::wstring AttributeNameUnpoolingWindowShape;
        CNTK_API static const std::wstring AttributeNameSubstitutionPenalty;
        CNTK_API static const std::wstring AttributeNameDeletionPenalty;
        CNTK_API static const std::wstring AttributeNameInsertionPenalty;
        CNTK_API static const std::wstring AttributeNameSquashInputs;
        CNTK_API static const std::wstring AttributeNameTokensToIgnore;
        CNTK_API static const std::wstring AttributeNameDelayConstraint;
        CNTK_API static const std::wstring AttributeNameBlankTokenId;
        CNTK_API static const std::wstring AttributeNamePhonePath;
        CNTK_API static const std::wstring AttributeNameSymListPath;
        CNTK_API static const std::wstring AttributeNameStateListPath;
        CNTK_API static const std::wstring AttributeNameTransProbPath;
        CNTK_API static const std::wstring AttributeNameLatticeConfigPath;
        CNTK_API static const std::wstring AttributeNameHSmoothingWeight;
        CNTK_API static const std::wstring AttributeNameFrameDropThresh;
        CNTK_API static const std::wstring AttributeNameDoReferenceAlign;
        CNTK_API static const std::wstring AttributeNameSeqGammarUsesMBR;
        CNTK_API static const std::wstring AttributeNameSeqGammarAMF;
        CNTK_API static const std::wstring AttributeNameSeqGammarLMF;
        CNTK_API static const std::wstring AttributeNameSeqGammarBMMIFactor;
        CNTK_API static const std::wstring AttributeNameSeqGammarWordPen;
        CNTK_API static const std::wstring AttributeNameNumClass;
        CNTK_API static const std::wstring AttributeNameOneHotOutputSparse;
        CNTK_API static const std::wstring AttributeNameOutputSparse;
        CNTK_API static const std::wstring AttributeNameOneHotAxis;
        CNTK_API static const std::wstring AttributeNameSequenceAxisNamePrefix;
        CNTK_API static const std::wstring AttributeNameSequenceUnpackPaddingValue;
        CNTK_API static const std::wstring AttributeNameSequenceUnpackSuppressMaskOutput;
        CNTK_API static const std::wstring AttributeNameRandomDistributionType;
        CNTK_API static const std::wstring AttributeNameRandomDistributionArgs;
        CNTK_API static const std::wstring AttributeNameSpatialScale;
        CNTK_API static const std::wstring AttributeNameSliceStrides;
        CNTK_API static const std::wstring AttributeNameSliceStridesVec;
        CNTK_API static const std::wstring AttributeNamePaddingHead;
        CNTK_API static const std::wstring AttributeNamePaddingFoot;
        CNTK_API static const std::wstring AttributeNamePaddingMode;
        CNTK_API static const std::wstring AttributeNamePaddingConstantValue;
        CNTK_API static const std::wstring AttributeNameAlpha;
        CNTK_API static const std::wstring AttributeNameBeta;
        CNTK_API static const std::wstring AttributeNameGamma;
        CNTK_API static const std::wstring AttributeNameKernelShape;
        CNTK_API static const std::wstring AttributeNameBias;
        CNTK_API static const std::wstring AttributeNameDepthRadius;
        CNTK_API static const std::wstring AttributeNameBlockSize;
        CNTK_API static const std::wstring AttributeNameCustomAttributes;
        CNTK_API static const std::wstring AttributeNameNumItems;
        CNTK_API static const std::wstring AttributeNameFillValue;
        CNTK_API static const std::wstring AttributeNameUseStatsAcrossChannels;
        CNTK_API static const std::wstring AttributeNameDoVarianceScaling;
        CNTK_API static const std::wstring AttributeNameGroups;
        CNTK_API static const std::wstring AttributeNameCustomOp;
        CNTK_API static const std::wstring AttributeNameTransposeLeftOperand;
        CNTK_API static const std::wstring AttributeNameTransposeRightOperand;

        CNTK_API static const std::vector<std::wstring> s_rngStateAttributes;
    };
}
