// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/graph/function.h"
#include "core/graph/graph_base.h"
#include "core/graph/model.h"

namespace onnxruntime {
class Graph;
class Node;
}  // namespace onnxruntime

namespace onnxruntime {

// Function representation class.
class FunctionImpl : public Function {
 public:
  FunctionImpl(const onnxruntime::Graph& graph,
               std::unique_ptr<IndexedSubGraph> customized_func);

  virtual const ONNX_NAMESPACE::OpSchema& OpSchema() const override;

  virtual const onnxruntime::GraphBase& Body() const override;

  virtual const IndexedSubGraph& GetIndexedSubGraph() const override;

 private:
  const onnxruntime::Graph* parent_graph_;
  std::unique_ptr<IndexedSubGraph> customized_func_body_;
  std::unique_ptr<ONNX_NAMESPACE::OpSchema> op_schema_;
  std::unique_ptr<onnxruntime::Model> body_;
};

}  // namespace onnxruntime
