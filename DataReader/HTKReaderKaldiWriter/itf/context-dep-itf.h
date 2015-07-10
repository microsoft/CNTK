// itf/context-dep-itf.h

// Copyright 2009-2011     Microsoft Corporation;  Go Vivace Inc.

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


#ifndef KALDI_ITF_CONTEXT_DEP_ITF_H_
#define KALDI_ITF_CONTEXT_DEP_ITF_H_
#include "base/kaldi-common.h"

namespace kaldi {
/// @ingroup tree_group
/// @{

/// context-dep-itf.h provides a link between
/// the tree-building code in ../tree/, and the FST code in ../fstext/
/// (particularly, ../fstext/context-dep.h).  It is an abstract
/// interface that describes an object that can map from a
/// phone-in-context to a sequence of integer leaf-ids.
class ContextDependencyInterface {
 public:
  /// ContextWidth() returns the value N (e.g. 3 for triphone models) that says how many phones
  ///   are considered for computing context.
  virtual int ContextWidth() const = 0;

  /// Central position P of the phone context, in 0-based numbering, e.g. P = 1 for typical
  /// triphone system.  We have to see if we can do without this function.
  virtual int CentralPosition() const = 0;

  /// The "new" Compute interface.  For typical topologies,
  /// pdf_class would be 0, 1, 2.
  /// Returns success or failure; outputs the pdf-id.
  ///
  /// "Compute" is the main function of this interface, that takes a
  /// sequence of N phones (and it must be N phones), possibly
  /// including epsilons (symbol id zero) but only at positions other
  /// than P [these represent unknown phone context due to end or
  /// begin of sequence].  We do not insist that Compute must always
  /// output (into stateseq) a nonempty sequence of states, but we
  /// anticipate that stateseq will alyway be nonempty at output in
  /// typical use cases.  "Compute" returns false if expansion somehow
  /// failed.  Normally the calling code should raise an exception if
  /// this happens.  We can define a different interface later in
  /// order to handle other kinds of information-- the underlying
  /// data-structures from event-map.h are very flexible.
  virtual bool Compute(const std::vector<int32> &phoneseq, int32 pdf_class,
                       int32 *pdf_id) const = 0;



  /// NumPdfs() returns the number of acoustic pdfs (they are numbered 0.. NumPdfs()-1).
  virtual int32 NumPdfs() const = 0;

  virtual ~ContextDependencyInterface() {};
  ContextDependencyInterface() {}

  /// Returns pointer to new object which is copy of current one.
  virtual ContextDependencyInterface *Copy() const = 0;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(ContextDependencyInterface);
};
/// @}
}  // namespace Kaldi


#endif
