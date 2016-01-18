//
// <copyright file="Actions.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

// This file represents the beginning of moving actions out from CNTK.cpp to make them accessible as a library. To be continued...

#include "Basics.h"
#include "Config.h"
#include "ScriptableObjects.h"
#include "DataReader.h"

// ===========================================================================
// implementations of all the commands of CNTK
// ===========================================================================

using namespace Microsoft::MSR::CNTK; // TODO: we should not have this in a header

// training (TrainActions.cpp)
template <class ConfigRecordType, typename ElemType>
void DoTrain(const ConfigRecordType& config);
template <typename ElemType>
void DoAdapt(const ConfigParameters& config);
template <typename ElemType>
void DoEdit(const ConfigParameters& config);

// evaluation (EvalActions.cpp)
template <typename ElemType>
void DoEval(const ConfigParameters& config);
template <typename ElemType>
void DoCrossValidate(const ConfigParameters& config);
template <typename ElemType>
void DoWriteOutput(const ConfigParameters& config);

// misc (OtherActions.cp)
template <typename ElemType>
void DoCreateLabelMap(const ConfigParameters& config);
template <typename ElemType>
void DoParameterSVD(const ConfigParameters& config);
template <typename ElemType>
void DoWriteWordAndClassInfo(const ConfigParameters& config);
template <typename ElemType>
void DoTopologyPlot(const ConfigParameters& config);

// deprecated (EsotericActions.cp)
template <typename ElemType>
void DoConvertFromDbn(const ConfigParameters& config);
template <typename ElemType>
void DoEvalUnroll(const ConfigParameters& config);
template <typename ElemType>
void DoEncoderDecoder(const ConfigParameters& config);
template <typename ElemType>
void DoBidirectionEncoderDecoder(const ConfigParameters& config);
template <typename ElemType>
void DoEvalEncodingBeamSearchDecoding(const ConfigParameters& config);
template <typename ElemType>
void DoBeamSearchDecoding(const ConfigParameters& config);
