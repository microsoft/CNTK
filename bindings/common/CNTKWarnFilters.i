//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKWarnFilters.i -- define warnfilters common for Python, C# and Java
//

// This file contains common warnfilters for Python, C# and Java

// Disabling warning about constructor shadowing, learner tests check this.
%warnfilter(401, 509) CNTK::TrainingParameterPerUnitSchedule;
%warnfilter(509) CNTK::MomentumAsTimeConstantSchedule;
%warnfilter(509) CNTK::NDArrayView::NDArrayView;

%warnfilter(315) CNTK::TrainingParameterPerSampleSchedule;

// Disabling warning about movable constructor shadowing, io tests check this.
%warnfilter(509) CNTK::DictionaryValue::DictionaryValue;
%warnfilter(509) CNTK::Dictionary::Dictionary;

// Disabling warning about Trainer shadowing, trainer tests check this.
%warnfilter(509) TrainerImpl;

// Returning an immutable string by reference.
%warnfilter(473) CNTK::Function::OpName;

// Specialization of non-template function - hash,
// TODO: it is not clear how to limit this only to hash, but we do not use partial specialization in other places.
#pragma SWIG nowarn=-317

// Disabling enable_shared_from_this - we never use this class to actually access the object.
%warnfilter(401) CNTK::NDArrayView;
%warnfilter(401) CNTK::NDMask;
%warnfilter(401) CNTK::Function;
%warnfilter(401) CNTK::Internal::UDFDeserializeCallbackWrapper;
%warnfilter(401) CNTK::Trainer;
%warnfilter(401) CNTK::Evaluator;
%warnfilter(401) CNTK::Value;
%warnfilter(401) CNTK::BackPropState;
%warnfilter(401) CNTK::MinibatchSource;

%warnfilter(401, 509) CNTK::MomentumAsTimeConstantSchedule;

%warnfilter(340) CNTK::NoOp;
