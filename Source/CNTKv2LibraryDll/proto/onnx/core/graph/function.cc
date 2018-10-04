// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/function_impl.h"
#include "core/graph/graph.h"
#include "core/graph/function_container.h"

namespace onnxruntime {

FunctionImpl::FunctionImpl(const onnxruntime::Graph& graph,
                           std::unique_ptr<IndexedSubGraph> customized_func) {
  parent_graph_ = &graph;
  customized_func_body_ = std::move(customized_func);
  auto meta_def = customized_func_body_->GetMetaDef();
  op_schema_ = std::make_unique<ONNX_NAMESPACE::OpSchema>();
  op_schema_->SetName(meta_def->name);
  op_schema_->SetDomain(meta_def->domain);
  op_schema_->SetDoc(meta_def->doc_string);
  op_schema_->SinceVersion(meta_def->since_version);
  int i = 0;
  for (auto& input : meta_def->inputs) {
    auto input_type = parent_graph_->GetNodeArg(input)->Type();
    op_schema_->Input(i, input, "", *input_type);
    ++i;
  }
  i = 0;
  for (auto& output : meta_def->outputs) {
    auto output_type = parent_graph_->GetNodeArg(output)->Type();
    op_schema_->Output(i, output, "", *output_type);
    ++i;
  }
  op_schema_->Finalize();
  //construct body
  std::unordered_map<std::string, int> domain_to_version;
  //TODO: set correct domain and version
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  body_ = std::make_unique<onnxruntime::Model>("fused_function_subgraph", false, onnxruntime::ModelMetaData(),
                                               /*TODO: get custom schema*/ nullptr, domain_to_version);

  auto& sub_graph = body_->MainGraph();
  //Add node and node args
  //TODO: for better performance, we could try to transfer the nodes in parent graph to sub-graph directly,
  //instead of create new nodes.
  for (auto& node_index : customized_func_body_->nodes) {
    auto node = parent_graph_->GetNode(node_index);
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    for (auto input : node->InputDefs()) {
      auto& n_input = sub_graph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
    }
    for (auto output : node->OutputDefs()) {
      auto& n_output = sub_graph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    sub_graph.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
  }
  //TODO: if we reuse the nodes in parent graph, maybe we don't need to resolve it.
  LOTUS_ENFORCE(sub_graph.Resolve().IsOK());
}

const ONNX_NAMESPACE::OpSchema& FunctionImpl::OpSchema() const {
  return *op_schema_;
}

const onnxruntime::GraphBase& FunctionImpl::Body() const {
  return body_->MainGraph();
}

const IndexedSubGraph& FunctionImpl::GetIndexedSubGraph() const {
  return *customized_func_body_;
}

std::unique_ptr<Function> MakeFunction(const onnxruntime::Graph& graph,
                                       std::unique_ptr<IndexedSubGraph> customized_func) {
  return std::make_unique<FunctionImpl>(graph, std::move(customized_func));
}
}  // namespace onnxruntime
