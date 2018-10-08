// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_base.h"

namespace onnxruntime {
class Function;
struct IndexedSubGraph;
}  // namespace onnxruntime

namespace onnxruntime {
struct FunctionContainer;
// A graph representation class.
class Graph : public GraphBase {
 public:
  // Resolve <*this> graph to ensure it's in a good shape with full
  // functionality.
  // 1. Run through all validation rules.
  //    a. Node name and node output's names should be unique.
  //    b. Attribute match between node and op definition.
  //    c. Input/Output match between node and op definition.
  //    d. Graph is acyclic and sort nodes in topological order.
  // 2. Check & Setup inner nodes' dependency.
  // 3. Cleanup function definition lists.
  // Returns resolving status.
  ::onnxruntime::common::Status Resolve() override;

  // Getter and Setter for graph name.
  const std::string& Name() const noexcept override;
  void SetName(const std::string& name) override;

  const std::string& Description() const noexcept override;
  void SetDescription(const std::string& description) override;

  // Add/Remove/Get initial tensors for some graph inputs.
  void AddInitializedTensor(const ONNX_NAMESPACE::TensorProto& tensor_proto);
  void RemoveInitializedTensor(const std::string& tensor_name);
  bool GetInitializedTensor(const std::string& tensor_name, gsl::not_null<const ONNX_NAMESPACE::TensorProto**> value) const;
  const InitializedTensorSet& GetAllInitializedTensors() const noexcept;
  void CleanAllInitializedTensors() noexcept;

  // Get graph value infos.
  const std::vector<const NodeArg*>& GetValueInfo() const noexcept;

  // Serialize the <Graph> into <GraphProto>.
  const ONNX_NAMESPACE::GraphProto& ToGraphProto();

  ILotusOpSchemaCollectionPtr GetSchemaRegistry() const;

  // Construct a Graph instance for a subgraph. Inherits some properties from the parent graph.
  Graph(const Graph& model_graph, ONNX_NAMESPACE::GraphProto& subgraph_proto);

  Node* FuseSubGraph(std::unique_ptr<::onnxruntime::IndexedSubGraph> sub_graph, const std::string& fused_node_name);

  void CollectRootNodesAndRefs();
  const std::vector<NodeIndex>& GetRootNodes() const { return root_nodes_; }
  const std::vector<size_t>& GetNodeRefs() const { return node_refs_; }

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(Graph);

  // This friendship relationship should only be used to call Graph::Graph and
  // Graph::LoadGraph All other access should be via the public API.
  friend class Model;

  // Constructor: Given a <GraphProto> loaded from model file, construct
  // a <Graph> object.
  Graph(ONNX_NAMESPACE::GraphProto* graph_proto,
        const std::unordered_map<std::string, int>& domain_to_version,
        Version ir_version,
        ILotusOpSchemaCollectionPtr schema_registry);

  Graph() = delete;

  enum class Type {
    // A main graph.
    Main = 1,
    // A sub graph (function).
    Sub = 2,
  };

  ::onnxruntime::common::Status Resolve(bool no_proto_sync_required);

  ::onnxruntime::common::Status InferAndVerifyTypeMatch(Node& node,
                                                        const ONNX_NAMESPACE::OpSchema& op);

  // Apply type-inference and type-checking to all inputs and initializers:
  ::onnxruntime::common::Status TypeCheckInputsAndInitializers();

  // Compute set of input and initializer names and checking for duplicate names
  ::onnxruntime::common::Status VerifyInputAndInitializerNames(
      /*OUT*/ std::unordered_set<std::string>& inputs_and_initializers);

  // Given nodes in topological order, infer and set type information
  // across <*this> graph if needed, and verify type/attribute
  // information match between node and op.
  ::onnxruntime::common::Status VerifyNodeAndOpMatch(const std::vector<NodeIndex>& nodes_in_topological_order,
                                                     const std::unordered_map<std::string, Node*>& output_args);

  void ComputeGraphInputsOutputsAndResetValues(std::vector<const NodeArg*> &new_graph_inputs,
      std::vector<const NodeArg*> &new_graph_outputs);

  // Set graph inputs/outputs when resolving a graph..
  ::onnxruntime::common::Status SetGraphInputsOutputs();

  // Sync graph inputs/outputs when serializing to proto.
  void SyncGraphInputsOutputs();

  // Clear all unused initializers
  void CleanUnusedInitializers();

  // GraphProto to store name, version, initializer.
  // When serializing <*this> Graph to a GraphProto, the nodes and
  // functions in <Graph> will also be fed into <graph_proto_> so that
  // it's consistent with <*this> graph.
  // This pointer is owned by parent model.
  ONNX_NAMESPACE::GraphProto* graph_proto_;

  // The node which refers to <*this> graph (Function).
  // Node* node_;

  std::unordered_map<std::string, int> name_to_initial_tensorIndex_;
  InitializedTensorSet name_to_initial_tensor_;
  std::vector<int> removed_initializer_indexes_;

  Type graph_type_ = Type::Main;

  // Graph value_info.
  std::vector<const NodeArg*> value_info_;

  ILotusOpSchemaCollectionPtr schema_registry_;

  std::unique_ptr<FunctionContainer> function_container_;

  std::vector<NodeIndex> root_nodes_;
  std::vector<size_t> node_refs_;
};
}  // namespace onnxruntime
