// itf/decodable-itf.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Mirko Hannemann;  Go Vivace Inc.

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_ITF_DECODABLE_ITF_H_
#define KALDI_ITF_DECODABLE_ITF_H_ 1
#include "base/kaldi-common.h"

namespace kaldi {
/// @ingroup Interfaces
/// @{

/// decodable-itf.h provides a link between the (acoustic-modeling and
/// feature-processing) code and the decoder.  The idea is to make this
/// interface as small as possible, and to make it as agnostic as possible about
/// the form of the acoustic model (e.g. don't assume the probabilities are a
/// function of just a vector of floats), and about the decoder (e.g. don't
/// assume it accesses frames in strict left-to-right order).  For normal
/// models, without on-line operation, the "decodable" sub-class will just be a
/// wrapper around a matrix of features and an acoustic model, and it will
/// answer the question 'what is the acoustic likelihood for this index and this
/// frame?'.

/// An interface for a feature-file and model; see \ref decodable_interface

class DecodableInterface {
 public:
  /// Returns the log likelihood, which will be negated in the decoder.
  virtual BaseFloat LogLikelihood(int32 frame, int32 index) = 0;

  /// Returns true if this is the last frame.  Frames are one-based.
  virtual bool IsLastFrame(int32 frame) = 0;

  // virtual int32 NumFrames() = 0;
  /// Returns the number of indices that the decodable object can accept;

  /// Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() = 0;
  
  virtual ~DecodableInterface() {}
};
/// @}
}  // namespace Kaldi

#endif  // KALDI_ITF_DECODABLE_ITF_H_
