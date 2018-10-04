// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_transformer_mgr.h"
using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

Status GraphTransformerManager::ApplyAll(Graph& graph) const {
  bool changed = false;
  for (unsigned step = 0; step < steps_; ++step) {
    for (auto& transformer : transformers_) {
      bool t_changed = false;
      Status s = transformer->Apply(graph, t_changed);
      if (!s.IsOK()) return s;
      changed = changed || t_changed;
    }
    if (!changed) break;
  }
  return Status::OK();
}

}  // namespace onnxruntime
