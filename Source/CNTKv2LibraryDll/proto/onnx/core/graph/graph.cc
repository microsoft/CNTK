// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _WIN32
// disable some warnings from protobuf to pass Windows build
#pragma warning(disable : 4244)
#endif

#include <fstream>
#include <iostream>
#include <numeric>
#include <stack>

#include "gsl/pointers"
#include "core/graph/function.h"
#include "core/graph/function_impl.h"
#include "core/graph/graph.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"
#include "onnx/checker.h"
#include "core/graph/schema_registry.h"
#include "core/graph/function_container.h"
using namespace ONNX_NAMESPACE;
using namespace ONNX_NAMESPACE::Utils;
using namespace ONNX_NAMESPACE::checker;
using namespace ::onnxruntime::common;

namespace onnxruntime {

#define NO_CHANGE_ON_SYNC_FLAG(...)                  \
  do {                                               \
    const bool sync_needed = GraphProtoSyncNeeded(); \
    { __VA_ARGS__; }                                 \
    GraphProtoSyncNeeded(sync_needed);               \
  } while (0)

static Status MergeShapeInfo(const std::string& output_name,
                             const TypeProto_Tensor& source, TypeProto_Tensor& target) {
  try {
    ONNX_NAMESPACE::mergeInShapeInfo(source, target);
  } catch (const ONNX_NAMESPACE::InferenceError& ex) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output:", output_name, " ", ex.what());
  }

  return Status::OK();
}

static bool GraphLoadedFromModelFile(const GraphProto* graph_proto) {
  return graph_proto && (graph_proto->input_size() != 0 ||
                         graph_proto->output_size() != 0 ||
                         graph_proto->value_info_size() != 0);
}

NodeArg::NodeArg(const std::string& name,
                 const TypeProto* p_node_arg_type) {
  node_arg_info_.set_name(name);
  // If the name is empty, it means the arg does not exist.
  exists_ = !(name.empty());
  if (nullptr != p_node_arg_type) {
    (*node_arg_info_.mutable_type()) = *p_node_arg_type;
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
  } else {
    type_ = nullptr;
  }
}

const std::string& NodeArg::Name() const noexcept {
  return node_arg_info_.name();
}

DataType NodeArg::Type() const noexcept {
  return type_;
}

const TypeProto* NodeArg::TypeAsProto() const noexcept {
  if (node_arg_info_.has_type())
    return &node_arg_info_.type();
  else
    return nullptr;
}

const TensorShapeProto* NodeArg::Shape() const {
  if (!node_arg_info_.has_type()) {
    return nullptr;
  }

  const auto typeCase = node_arg_info_.type().value_case();
  switch (typeCase) {
    case TypeProto::kTensorType: {
      if (node_arg_info_.type().tensor_type().has_shape()) {
        return &(node_arg_info_.type().tensor_type().shape());
      } else {
        return nullptr;
      }
    }
    case TypeProto::kSparseTensorType: {
      if (node_arg_info_.type().sparse_tensor_type().has_shape()) {
        return &(node_arg_info_.type().sparse_tensor_type().shape());
      } else {
        return nullptr;
      }
    }
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return nullptr;
  }
}

void NodeArg::SetShape(const TensorShapeProto& shape) {
  if (!node_arg_info_.has_type()) {
    return;
  }

  const auto type_case = node_arg_info_.type().value_case();
  switch (type_case) {
    case TypeProto::kTensorType:
      *(node_arg_info_.mutable_type()->mutable_tensor_type()->mutable_shape()) = shape;
      break;
    case TypeProto::kSparseTensorType:
      *(node_arg_info_.mutable_type()->mutable_sparse_tensor_type()->mutable_shape()) = shape;
      break;
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
    default:
      return;
  }
}

common::Status NodeArg::UpdateTypeAndShape(const ONNX_NAMESPACE::TypeProto& input_type) {
  if (!node_arg_info_.has_type()) {
    *node_arg_info_.mutable_type() = input_type;
    type_ = DataTypeUtils::ToType(node_arg_info_.type());
    return Status::OK();
  }

  auto& current_type = *node_arg_info_.mutable_type();
  const auto current_type_case = current_type.value_case();
  const auto input_type_case = input_type.value_case();

  if (current_type_case != input_type_case)
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type mismatch. Current=",
                                   current_type_case, " Input=", input_type_case);

  switch (input_type_case) {
    case TypeProto::kTensorType: {
      const auto& input_tensor_type = input_type.tensor_type();
      const auto& input_tensor_elem_type = input_tensor_type.elem_type();
      const auto& current_tensor_elem_type = current_type.tensor_type().elem_type();

      if (input_tensor_elem_type != current_tensor_elem_type)
        return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Tensor element type mismatch. ",
                                       TensorProto_DataType_Name(input_tensor_elem_type), " != ",
                                       TensorProto_DataType_Name(current_tensor_elem_type));

      if (input_tensor_type.has_shape()) {
        auto& current_tensor_type = *current_type.mutable_tensor_type();
        if (current_tensor_type.has_shape()) {
          ONNXRUNTIME_RETURN_IF_ERROR(MergeShapeInfo(Name(), input_tensor_type, current_tensor_type));
        } else {
          current_tensor_type = input_tensor_type;
        }
      }

      break;
    }
    case TypeProto::kSparseTensorType: {
      const auto& input_tensor_type = input_type.sparse_tensor_type();
      const auto input_tensor_elem_type = input_tensor_type.elem_type();
      const auto current_tensor_elem_type = current_type.sparse_tensor_type().elem_type();
      if (input_tensor_elem_type != current_tensor_elem_type) {
        return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "SparseTensor element type mismatch. ",
                                       TensorProto_DataType_Name(input_tensor_elem_type), " != ",
                                       TensorProto_DataType_Name(current_tensor_elem_type));
      }
      if (input_tensor_type.has_shape()) {
        auto& current_tensor_type = *current_type.mutable_sparse_tensor_type();
        if (current_tensor_type.has_shape()) {
          // TODO: Check if we need to merge shape here
          // if so we'd need to provide merging routine ONNX
          // mergeInShapeInfo(input_tensor_type, current_tensor_type);
        } else {
          current_tensor_type = input_tensor_type;
        }
      }
    } break;
    case TypeProto::kSequenceType:
    case TypeProto::kMapType:
    case TypeProto::kOpaqueType:
    case TypeProto::VALUE_NOT_SET:
      break;
  }

  return Status::OK();
}

common::Status NodeArg::UpdateTypeAndShape(const NodeArg& node_arg) {
  auto status = Status::OK();

  if (node_arg.node_arg_info_.has_type())
    status = UpdateTypeAndShape(node_arg.node_arg_info_.type());

  return status;
}

void NodeArg::SetType(DataType p_type) {
  if (nullptr == p_type) {
    return;
  }

  type_ = p_type;
  *(node_arg_info_.mutable_type()) = DataTypeUtils::ToTypeProto(p_type);
}

void NodeArg::SetType(const TypeProto& type_proto) {
  type_ = DataTypeUtils::ToType(type_proto);
  *(node_arg_info_.mutable_type()) = type_proto;
}

bool NodeArg::Exists() const noexcept {
  return exists_;
}

Node::EdgeEnd::EdgeEnd(const Node& node, const NodeArg& node_arg) noexcept
    : node_(&node), node_arg_(&node_arg) {
}

Node::EdgeEnd::EdgeEnd(const Node& node) noexcept
    : node_(&node), node_arg_(nullptr) {
}

const Node& Node::EdgeEnd::GetNode() const noexcept {
  return *node_;
}

const NodeArg* Node::EdgeEnd::GetNodeArg() const noexcept {
  return node_arg_;
}

Node::NodeConstIterator::NodeConstIterator(EdgeConstIterator p_iter) {
  m_iter = p_iter;
}

bool Node::NodeConstIterator::operator==(const NodeConstIterator& p_other) const {
  return m_iter == p_other.m_iter;
}

bool Node::NodeConstIterator::operator!=(const NodeConstIterator& p_other) const {
  return m_iter != p_other.m_iter;
}

void Node::NodeConstIterator::operator++() {
  ++m_iter;
}

void Node::NodeConstIterator::operator--() {
  --m_iter;
}

const Node* Node::NodeConstIterator::operator*() {
  return &((*m_iter).GetNode());
}

NodeIndex Node::Index() const noexcept {
  return index_;
}

const std::string& Node::Name() const noexcept {
  return name_;
}

const std::string& Node::OpType() const noexcept {
  return op_type_;
}

const std::string& Node::Description() const noexcept {
  return description_;
}

const std::string& Node::Domain() const noexcept {
  return domain_;
}

const OpSchema* Node::Op() const noexcept {
  return op_;
}

Node::Type Node::NodeType() const noexcept {
  return node_type_;
}

void Node::SetNodeType(Node::Type node_type) noexcept {
  node_type_ = node_type;
}

const ::onnxruntime::Function* Node::GetFunctionBody() const noexcept {
  return func_body_;
}

void Node::SetFunctionBody(const ::onnxruntime::Function& func) {
  func_body_ = &func;
  op_ = &func.OpSchema();
}

const std::string& Node::GetExecutionProviderType() const noexcept {
  return execution_provider_type_;
}

void Node::SetExecutionProviderType(onnxruntime::ProviderType execution_provider_type) {
  execution_provider_type_ = execution_provider_type;
}

void Node::ToProto(NodeProto& proto) const {
  // Set name.
  proto.set_name(name_);
  // Set op type.
  proto.set_op_type(op_type_);
  // Set op domain;
  proto.set_domain(domain_);
  // Set doc string.
  proto.set_doc_string(description_);

  // Set attributes.
  proto.clear_attribute();
  for (auto attribute : attributes_) {
    const gsl::not_null<AttributeProto*> attr{proto.add_attribute()};
    *attr = attribute.second;
  }

  // Set inputs' definitions.
  proto.clear_input();
  for (auto& input_def : definitions_.input_defs) {
    proto.add_input(input_def->Name());
  }

  // Set outputs' definitions.
  proto.clear_output();
  for (auto& output_def : definitions_.output_defs) {
    proto.add_output(output_def->Name());
  }
}

void Node::Init(const std::string& name,
                const std::string& op_type,
                const std::string& description,
                const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                const NodeAttributes* attributes,
                const std::string& domain) {
  name_ = name;
  op_type_ = op_type;
  description_ = description;
  definitions_.input_defs = input_args;
  definitions_.output_defs = output_args;
  domain_ = domain;
  if (kOnnxDomainAlias == domain_) {
    domain_ = kOnnxDomain;
  }

  // Set each arg count as 1 by default.
  // It could be adjusted when resolving the node with its operator
  // information.
  definitions_.input_arg_count.assign(input_args.size(), 1);

  if (attributes) {
    attributes_ = *attributes;
  }
}

Node::Definitions& Node::MutableDefinitions() noexcept {
  // someone fetching these is going to change something
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return definitions_;
}

Node::Relationships& Node::MutableRelationships() noexcept {
  // someone fetching these is going to change something
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return relationships_;
}

void Node::AddAttribute(const std::string& attr_name, const AttributeProto& value) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  attributes_[attr_name] = value;
}

#define ADD_BASIC_ATTR_IMPL(type, enumType, field)                           \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->SetGraphResolveNeeded();                                         \
    graph_->SetGraphProtoSyncNeeded();                                       \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    a.set_type(enumType);                                                    \
    a.set_##field(value);                                                    \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_ATTR_IMPL(type, enumType, field)                                 \
  void Node::AddAttribute(const std::string& attr_name, const type& value) { \
    graph_->SetGraphResolveNeeded();                                         \
    graph_->SetGraphProtoSyncNeeded();                                       \
    AttributeProto a;                                                        \
    a.set_name(attr_name);                                                   \
    a.set_type(enumType);                                                    \
    *(a.mutable_##field()) = value;                                          \
    attributes_[attr_name] = a;                                              \
  };

#define ADD_LIST_ATTR_IMPL(type, enumType, field)            \
  void Node::AddAttribute(const std::string& attr_name,      \
                          const std::vector<type>& values) { \
    graph_->SetGraphResolveNeeded();                         \
    graph_->SetGraphProtoSyncNeeded();                       \
    AttributeProto a;                                        \
    a.set_name(attr_name);                                   \
    a.set_type(enumType);                                    \
    for (const auto& val : values) {                         \
      *(a.mutable_##field()->Add()) = val;                   \
    }                                                        \
    attributes_[attr_name] = a;                              \
  };

ADD_BASIC_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT, f)
ADD_BASIC_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INT, i)
ADD_BASIC_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRING, s)
ADD_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR, t)
ADD_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH, g)
ADD_LIST_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS, floats)
ADD_LIST_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INTS, ints)
ADD_LIST_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS, strings)
ADD_LIST_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS, tensors)
ADD_LIST_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPHS, graphs)

bool Node::ClearAttribute(const std::string& attr_name) {
  graph_->SetGraphResolveNeeded();
  graph_->SetGraphProtoSyncNeeded();
  return attributes_.erase(attr_name) > 0;
}

Status Node::UpdateInputArgCount() {
  // The node refers to a primitive operator.
  // Infer and verify node input arg type information.
  int total_arg_count = std::accumulate(definitions_.input_arg_count.cbegin(),
                                        definitions_.input_arg_count.cend(), 0);

  if (total_arg_count < 0 || static_cast<size_t>(total_arg_count) != definitions_.input_defs.size()) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "The sum of input arg count is not equal to size of input defs in node (",
                                   name_, ")");
  }

  // op_ is always valid when this is called
  const ONNX_NAMESPACE::OpSchema& op = *Op();

  // Verify size of node arg count is same as input number in
  // operator definition.
  if (op.inputs().size() != definitions_.input_arg_count.size()) {
    // Adjust input arg count array with op definition
    // The adjustment will work as below,
    // In total, there're <total_arg_count> inputs, which
    // will be split as <1, 1, 1, 1, ... 1, x> or
    // <1, 1, 1, 1, ...1, 0, 0, ...0>. The final input
    // arg count array's element number will be the same
    // as op definition, and the sum of all elements will
    // be equal to <total_arg_count>.
    auto& input_arg_count = definitions_.input_arg_count;
    input_arg_count.clear();
    size_t m = 0;
    auto arg_count_left = total_arg_count;

    if (!op.inputs().empty()) {
      for (; m < op.inputs().size() - 1; ++m) {
        if (arg_count_left > 0) {
          input_arg_count.push_back(1);
          arg_count_left--;
        } else {
          input_arg_count.push_back(0);
        }
      }
    }

    // Set the arg count for the last input formal parameter.
    // NOTE: in the case that there's no .input(...) defined
    // in op schema, all input args will be fed as one input
    // of the operator.
    input_arg_count.push_back(arg_count_left);

    graph_->SetGraphResolveNeeded();
    graph_->SetGraphProtoSyncNeeded();
  }

  return Status::OK();
}

const NodeAttributes& Node::GetAttributes() const noexcept {
  return attributes_;
}

void Node::ForEachDef(std::function<void(const onnxruntime::NodeArg*, bool is_input)> func) const {
  for (const auto* arg : InputDefs()) {
    if (arg->Exists())
      func(&*arg, true);
  }

  for (const auto* arg : ImplicitInputDefs()) {
    if (arg->Exists())
      func(&*arg, true);
  }

  for (const auto* arg : OutputDefs()) {
    if (arg->Exists())
      func(&*arg, false);
  }
};

void Node::ForEachInputDef(std::function<void(const onnxruntime::NodeArg*)> func) const {
  for (const auto* arg : InputDefs()) {
    if (!arg->Exists())
      continue;
    func(&*arg);
  }
};

void Node::ForEachOutputDef(std::function<void(const onnxruntime::NodeArg*)> func) const {
  for (const auto* arg : OutputDefs()) {
    if (!arg->Exists())
      continue;
    func(&*arg);
  }
};

void Node::ReplaceDefs(const std::map<const onnxruntime::NodeArg*, onnxruntime::NodeArg*>& replacements) {
  std::vector<std::vector<NodeArg*>*> all_defs = {&definitions_.input_defs, &definitions_.output_defs};

  for (auto pair : replacements)
    for (auto* defs : all_defs)
      for (auto& def : *defs)
        if (def == pair.first)
          def = pair.second;
}

// Constructor: Given a <GraphProto> loaded from model file, construct
// a <Graph> object and Resolve() it.
//Status Graph::LoadGraph(const GraphProto& graph_proto,
//                        const std::unordered_map<std::string, int>& domain_to_version,
//                        Version ir_version,
//                        std::unique_ptr<Graph>& new_graph) {
//  // create instance. need to call private ctor so can't use make_unique
//  GSL_SUPPRESS(r .11)
//  new_graph.reset(new Graph(nullptr, &graph_proto, domain_to_version, ir_version));
//
//  // as we just loaded from file we want to fully initialize/Resolve, but not let that change
//  // the proto sync flag
//  auto status = new_graph->Resolve(/* no_proto_sync_required */ true);
//  return status;
//}
using google::protobuf::RepeatedPtrField;

Graph::Graph(GraphProto* graph_proto,
             const std::unordered_map<std::string, int>& domain_to_version,
             Version ir_version,
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry)
    : Graph(graph_proto, domain_to_version, ir_version, schema_registry, nullptr) {}

Graph::Graph(GraphProto* graph_proto,
             const std::unordered_map<std::string, int>& domain_to_version,
             Version ir_version,
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
             Graph* parent_graph)
    : graph_proto_{graph_proto},
      graph_type_{Type::Main},
      schema_registry_(schema_registry),
      function_container_(std::make_unique<FunctionContainer>()),
      graph_resolve_needed_(true),
      graph_proto_sync_needed_(false),
      domain_to_version_(domain_to_version),
      ir_version_(ir_version),
      parent_graph_{parent_graph} {
  ONNXRUNTIME_ENFORCE(graph_proto != nullptr, "graph_proto cannot be null");
  ArgNameToTypeMap name_to_type_map;

  // these are all empty unless we received a graph_proto as input
  if (graph_proto != nullptr) {
    // Copy constant nodes _value to name_to_initial_tensor_
    for (auto& node : graph_proto_->node()) {
      if (node.op_type() == kConstant) {
        const gsl::not_null<TensorProto*> tensor{graph_proto_->add_initializer()};
        *tensor = node.attribute(0).t();
        *(tensor->mutable_name()) = node.output(0);

        // we remove the node and add it as an initializer, but still need it to appear in the
        // graph inputs to make the ONNX checker happy. add a new input due to that.
        auto graph_inputs = graph_proto_->mutable_input();

        ValueInfoProto* value_info = graph_inputs->Add();
        value_info->set_name(node.output(0));
        value_info->set_doc_string("Input to represent replaced Constant node");

        TypeProto t;
        t.mutable_tensor_type()->set_elem_type(tensor->data_type());
        auto shape = t.mutable_tensor_type()->mutable_shape();
        for (auto dim : tensor->dims())
          shape->add_dim()->set_dim_value(dim);

        (*value_info->mutable_type()) = t;
      }
    }

    // remove constant nodes
    const gsl::not_null<RepeatedPtrField<NodeProto>*> graph_mutable_nodes{graph_proto_->mutable_node()};
    graph_mutable_nodes->erase(
        std::remove_if(graph_mutable_nodes->begin(), graph_mutable_nodes->end(),
                       [](NodeProto& p) {
                         return (p.op_type() == kConstant);
                       }),
        graph_mutable_nodes->end());

    // Copy initial tensors to a map.
    for (auto& tensor : graph_proto_->initializer()) {
      name_to_initial_tensor_[tensor.name()] = &tensor;
    }

    // Collect all node arg name, type, shape information in the graph.
    // type/shape information will be assigned to each node arg when going
    // thru all nodes later.
    for (auto& graph_input : graph_proto_->input()) {
      if (graph_input.has_name() && graph_input.has_type()) {
        name_to_type_map[graph_input.name()] = graph_input.type();
        // always create a NodeArg for graph input in case its from an initializer
        GetOrCreateNodeArg(graph_input.name(), &graph_input.type());
      }
    }

    for (auto& graph_output : graph_proto_->output()) {
      if (graph_output.has_name() && graph_output.has_type()) {
        auto& name = graph_output.name();
        name_to_type_map[name] = graph_output.type();
        // always create NodeArg for graph output, in case it's from initializer
        GetOrCreateNodeArg(name, &graph_output.type());
      }
    }

    for (auto& node_arg : graph_proto_->value_info()) {
      if (node_arg.has_name() && node_arg.has_type()) {
        name_to_type_map[node_arg.name()] = node_arg.type();
      }
    }

    for (auto node_proto : graph_proto_->node()) {
      AddNode(node_proto, name_to_type_map);
    }
  }
}

Graph::Graph(Graph& parent_graph, ONNX_NAMESPACE::GraphProto& subgraph_proto)
    : Graph(&subgraph_proto,
            parent_graph.DomainToVersionMap(), parent_graph.IrVersion(), parent_graph.schema_registry_,
            &parent_graph) {
}

Status Graph::VerifyNoDuplicateName() {
  const std::unordered_set<std::string>& inputs_and_initializers = resolve_context_.inputs_and_initializers;
  std::unordered_map<std::string, Node*>& output_args = resolve_context_.output_args;
  std::unordered_map<std::string, NodeIndex>& node_name_to_index = resolve_context_.node_name_to_index;

  output_args.clear();
  node_name_to_index.clear();
  // inputs_and_initializers: this is passed in as a parameter, since functions don't have initializers
  // but graphs have them.

  for (auto& node : Nodes()) {
    // Verify node name should be unique.
    auto& node_name = node.Name();

    if (!node_name.empty() && node_name_to_index.end() != node_name_to_index.find(node_name)) {
      // The node has name and its name was used by another node.
      Status status(ONNXRUNTIME, FAIL,
                    "Error: two nodes with same node name (" + node_name + ").");
      return status;
    }

    node_name_to_index[node_name] = node.Index();

    // Verify node outputs' name should be unique.
    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        auto& output_arg_name = output_def->Name();
        if (inputs_and_initializers.count(output_arg_name)) {
          Status status(ONNXRUNTIME, FAIL,
                        "Error: Duplicate definition of name (" + output_arg_name + ").");
          return status;
        }
        auto result = output_args.insert({output_arg_name, &node});
        if (!result.second) {
          // Two outputs with same name, so that insertion fails.
          Status status(ONNXRUNTIME, FAIL,
                        "Error: Duplicate definition of name (" + output_arg_name + ").");
          return status;
        }
      }
    }
  }
  return Status::OK();
}

// Recurse into any subgraphs to update the list of NodeArg values in outer scope.
// This information is needed to resolve any dependencies on outer scope values.
common::Status Graph::SetOuterScopeNodeArgs(const std::unordered_set<std::string>& outer_scope_node_args) {
  resolve_context_.outer_scope_node_args = outer_scope_node_args;

  if (!resolve_context_.node_to_subgraphs_map.empty()) {
    // Build the list of NodeArg's that are valid for a subgraph of this GraphBase instance:
    //   - outer scope for this graph
    //   - any inputs/initializers from this graph
    //   - any outputs from nodes in this graph
    //
    // NOTE: We must add the most outer most NodeArgs first, and then local NodeArgs, as the local should override
    // an outer scope value if they have the same name.
    //
    // We provide outputs from all nodes in this graph at this stage.
    // BuildConnections will link the node with the subgraph to any outer scope Node/NodeArgs it consumes.
    // PerformTopologicalSortAndCheckIsAcyclic will validate these links.
    std::unordered_set<std::string> node_args_in_scope_for_subgraph = outer_scope_node_args;

    node_args_in_scope_for_subgraph.insert(resolve_context_.inputs_and_initializers.cbegin(),
                                           resolve_context_.inputs_and_initializers.cend());

    std::transform(resolve_context_.output_args.cbegin(), resolve_context_.output_args.cend(),
                   std::inserter(node_args_in_scope_for_subgraph, node_args_in_scope_for_subgraph.end()),
                   [](const std::pair<std::string, Node*>& entry) { return entry.first; });

    for (auto node_subgraphs : resolve_context_.node_to_subgraphs_map) {
      for (auto* subgraph : node_subgraphs.second) {
        auto status = subgraph->SetOuterScopeNodeArgs(node_args_in_scope_for_subgraph);
        ONNXRUNTIME_RETURN_IF_ERROR(status);
      }
    }
  }

  return Status::OK();
}

const NodeArg* Graph::GetNodeArgIncludingParentGraphs(const std::string& node_arg_name) const {
  const NodeArg* node_arg = GetNodeArg(node_arg_name);

  if (!node_arg && parent_graph_) {
    node_arg = parent_graph_->GetNodeArgIncludingParentGraphs(node_arg_name);
  }

  return node_arg;
}

void Graph::AddEdge(NodeIndex src_node_index, NodeIndex dst_node_index, const NodeArg& node_arg) {
  if (nodes_.size() <= src_node_index ||
      nodes_.size() <= dst_node_index ||
      nullptr == nodes_[src_node_index] ||
      nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    ONNXRUNTIME_THROW("Invalid node indexes specified when adding edge.");
  }
  // Verify whether the node_arg is input of dst and output of src firstly.
  bool valid = false;
  for (auto arg : nodes_[src_node_index]->OutputDefs()) {
    if (arg == &node_arg) {
      valid = true;
      break;
    }
  }
  ONNXRUNTIME_ENFORCE(valid);
  valid = false;
  for (auto arg : nodes_[dst_node_index]->InputDefs()) {
    if (arg == &node_arg) {
      valid = true;
      break;
    }
  }
  for (auto arg : nodes_[dst_node_index]->ImplicitInputDefs()) {
    if (arg == &node_arg) {
      valid = true;
      break;
    }
  }
  ONNXRUNTIME_ENFORCE(valid);
  nodes_[dst_node_index]->MutableRelationships().input_edges.insert(Node::EdgeEnd(*nodes_[src_node_index], node_arg));
  nodes_[src_node_index]->MutableRelationships().output_edges.insert(Node::EdgeEnd(*nodes_[dst_node_index], node_arg));
}

void Graph::RemoveEdge(NodeIndex src_node_index, NodeIndex dst_node_index, const NodeArg& node_arg) {
  if (nodes_.size() <= src_node_index ||
      nodes_.size() <= dst_node_index ||
      nullptr == nodes_[src_node_index] ||
      nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    ONNXRUNTIME_THROW("Invalid node indexes specified when removing edge.");
  }
  // Verify whether the node_arg is input of dst and output of src firstly.
  bool valid = false;
  for (auto arg : nodes_[src_node_index]->OutputDefs()) {
    if (arg == &node_arg) {
      valid = true;
      break;
    }
  }
  ONNXRUNTIME_ENFORCE(valid);
  valid = false;
  for (auto arg : nodes_[dst_node_index]->InputDefs()) {
    if (arg == &node_arg) {
      valid = true;
      break;
    }
  }
  for (auto arg : nodes_[dst_node_index]->ImplicitInputDefs()) {
    if (arg == &node_arg) {
      valid = true;
      break;
    }
  }
  ONNXRUNTIME_ENFORCE(valid);
  nodes_[dst_node_index]->MutableRelationships().input_edges.erase(Node::EdgeEnd(*nodes_[src_node_index], node_arg));
  nodes_[src_node_index]->MutableRelationships().output_edges.erase(Node::EdgeEnd(*nodes_[dst_node_index], node_arg));
}

GSL_SUPPRESS(es .84)  // ignoring return value from unordered_map::insert causes noisy complaint
Status Graph::BuildConnections(std::vector<std::string>& outer_scope_node_args_consumed) {
  const std::unordered_set<std::string>& outer_scope_node_args = resolve_context_.outer_scope_node_args;
  std::unordered_set<Node*> inner_nodes;

  // recurse into subgraphs first so we can update any nodes in this graph that are used by those subgraphs
  if (!resolve_context_.node_to_subgraphs_map.empty()) {
    for (auto nodeid_to_subgraphs : resolve_context_.node_to_subgraphs_map) {
      for (auto* subgraph : nodeid_to_subgraphs.second) {
        std::vector<std::string> node_args_consumed;
        subgraph->BuildConnections(node_args_consumed);

        for (auto& node_arg_name : node_args_consumed) {
          const auto* node_arg = GetNodeArg(node_arg_name);

          if (node_arg == nullptr) {
            // it's a node arg from outside this graph's scope, so add that to the list we return
            // so that we can add the dependency at the next level up
              if (node_arg_name == "ElementTimes1147_Output_0")
                  std::cout << "";
            outer_scope_node_args_consumed.push_back(node_arg_name);

            if (!parent_graph_) {
              return ONNXRUNTIME_MAKE_STATUS(
                  ONNXRUNTIME, INVALID_GRAPH,
                  "At top level graph without matching NodeArg that subgraph consumes. Name=",
                  node_arg_name,
                  " Graph may not conform to the ONNX spec and contain initializers that are not graph inputs.");
            }

            node_arg = parent_graph_->GetNodeArgIncludingParentGraphs(node_arg_name);

            if (!node_arg) {
              return ONNXRUNTIME_MAKE_STATUS(
                  ONNXRUNTIME, INVALID_GRAPH,
                  "Failed to find NodeArg in all parent graphs. Name=", node_arg_name,
                  " Graph may not conform to the ONNX spec and contain initializers that are not graph inputs.");
            }
          }

          // add it to the Node's list of implicit inputs
          auto& node = *GetNode(nodeid_to_subgraphs.first);

          if (node_arg->Name() == "ElementTimes1147_Output_0")
              std::cout << "";
          node.MutableDefinitions().implicit_input_defs.push_back(node_arg);

          if (resolve_context_.inputs_and_initializers.find(node_arg_name) !=
              resolve_context_.inputs_and_initializers.cend()) {
            // no connection required
          } else {
            // if it's an output nodearg in this graph we need to create a link to the node the output is coming from
            auto entry = resolve_context_.output_args.find(node_arg_name);
            ONNXRUNTIME_ENFORCE(entry != resolve_context_.output_args.end());

            // Create relationship between this node (node), and the node providing the output (output_node).
            Node& output_node = *entry->second;
            AddEdge(output_node.Index(), node.Index(), *node_arg);

            inner_nodes.insert(&output_node);
          }
        }
      }
    }
  }

  // now build connections within this Graph instance
  for (auto& node : Nodes()) {
    // Need mutable input defs to be able to set any outer scope NodeArg implicit inputs
    auto& input_args = node.MutableInputDefs();

    if (input_args.size() > 0) {
      // This node needs inputs.

      for (const auto* input_arg : input_args) {
        if (!input_arg->Exists()) {
          // This input could be optional and it does not exist in this case.
          continue;
        }

        auto output_arg_iter = resolve_context_.output_args.find(input_arg->Name());
        if (resolve_context_.output_args.end() == output_arg_iter) {
          // No such output_arg matching this input_arg.
          // This input arg should be fed when running evaluation.
          // See if it's present in the outer scope. If so it will be 'fed' by the execution frame
          // providing access to the MLValue from the outer scope. Pass the name back up so nodes can
          // be linked correctly at that level.
          if (outer_scope_node_args.find(input_arg->Name()) != outer_scope_node_args.cend()) {
              if (input_arg->Name() == "ElementTimes1147_Output_0")
                  std::cout << "";

            outer_scope_node_args_consumed.push_back(input_arg->Name());
          }

          continue;
        }

        // Create relationship between this node (node), and the node providing the output (output_node).
        Node& output_node = *output_arg_iter->second;
        AddEdge(output_node.Index(), node.Index(), *input_arg);

        inner_nodes.insert(&output_node);
      }
    } else if (node.OutputDefs().size() <= 0) {
      // This is a useless node.
      // It has no input/output.
      RemoveNode(node.Index());
    }
  }

  return Status::OK();
}

void Graph::ReverseDFSFrom(const std::vector<NodeIndex>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) const {
  std::vector<const Node*> node_vec;
  for (auto i : from) {
    node_vec.push_back(GetNode(i));
  }

  ReverseDFSFrom(node_vec, enter, leave, comp);
}

void Graph::ReverseDFSFrom(const std::vector<const Node*>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) const {
  using WorkEntry = std::pair<const Node*, bool>;  // bool represents leave or not
  std::vector<WorkEntry> stack(from.size());
  for (size_t i = 0; i < from.size(); i++) {
    stack[i] = WorkEntry(from[i], false);
  }

  std::vector<bool> visited(MaxNodeIndex(), false);
  while (!stack.empty()) {
    const WorkEntry last_entry = stack.back();
    stack.pop_back();
    const Node& n = *last_entry.first;
    if (last_entry.second) {
      // leave node
      leave(&n);
      continue;
    }

    if (visited[n.Index()]) continue;

    visited[n.Index()] = true;

    if (enter) enter(&n);

    if (leave) stack.emplace_back(&n, true);

    if (comp) {
      std::vector<const Node*> sorted_nodes;
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        sorted_nodes.push_back((*iter));
      }
      std::sort(sorted_nodes.begin(), sorted_nodes.end(), comp);
      for (const auto* in : sorted_nodes) {
        const NodeIndex idx = in->Index();
        if (!visited[idx]) {
          stack.emplace_back(in, false);
        }
      }
    } else {
      for (auto iter = n.InputNodesBegin(); iter != n.InputNodesEnd(); ++iter) {
        const NodeIndex idx = (*iter)->Index();
        if (!visited[idx]) {
          stack.emplace_back(GetNode(idx), false);
        }
      }
    }
  }
}

GSL_SUPPRESS(es .84)  // noisy warning about ignoring return value from insert(...)
Status Graph::PerformTopologicalSortAndCheckIsAcyclic() {
  nodes_in_topological_order_.clear();
  // nodes that have been processed and added to nodes_in_topological_order.
  std::unordered_set<NodeIndex> processed_nodes;
  std::unordered_set<NodeIndex> output_nodes;
  std::unordered_set<NodeIndex> nodes_added_for_processing;
  std::stack<NodeIndex> stack;

  // push the top level nodes into nodes_in_topological_order in the order they were added
  // to ensure that is consistent.
  auto& nodes_in_original_order = Nodes();
  std::for_each(nodes_in_original_order.cbegin(), nodes_in_original_order.cend(),
                [&](const Node& node) {
                  auto index = node.Index();

                  // find the top level nodes in the graph.
                  // need to also consider nodes that only have Constants as inputs as top level nodes,
                  // as the constant will get replaced by an initializer.
                  auto input_edges = node.GetRelationships().input_edges;
                  auto has_inputs = std::any_of(input_edges.cbegin(), input_edges.cend(), [](const Node::EdgeEnd& edge) {
                    return edge.GetNode().OpType() != kConstant;
                  });

                  if (!has_inputs) {
                    // add to the topological list, and ensure we skip these nodes when walking the graph
                    nodes_in_topological_order_.push_back(index);
                    processed_nodes.insert(index);

                    // mark this as added as we've fully processed it and don't need to do it again later
                    nodes_added_for_processing.insert(index);
                  }
                });

  // start at the bottom and work our way up the graph
  for (auto iter = Nodes().begin(); iter != Nodes().end(); ++iter) {
    if (0 == iter->relationships_.output_edges.size()) {
      // This is a leaf node.
      stack.push(iter->Index());
    }
  }

  while (!stack.empty()) {
    const NodeIndex current = stack.top();
    stack.pop();

    if (processed_nodes.find(current) != processed_nodes.end()) {
      continue;
    }

    if (nodes_added_for_processing.find(current) != nodes_added_for_processing.end()) {
      // we popped the stack and are back to a node that was added previously,
      // so we know all the upstream nodes from it have been fully processed,
      nodes_in_topological_order_.push_back(current);
      processed_nodes.insert(current);
      output_nodes.erase(current);
      continue;
    }

    const Node* node = GetNode(current);
    if (!node) {
      continue;
    }

    stack.push(current);
    output_nodes.insert(current);

    for (auto iter = node->InputNodesBegin(); iter != node->InputNodesEnd(); ++iter) {
      const NodeIndex idx = (*iter)->Index();
      if (output_nodes.find(idx) != output_nodes.end()) {
        Status status(ONNXRUNTIME, FAIL, "Error: the graph is not acyclic.");
        return status;
      }

      // avoid re-processing nodes
      if (nodes_added_for_processing.find(idx) == nodes_added_for_processing.end()) {
        stack.push(idx);
      }
    }

    nodes_added_for_processing.insert(current);
  }

  if (num_of_nodes_ >= 0 && static_cast<size_t>(num_of_nodes_) == nodes_in_topological_order_.size()) {
    return Status::OK();
  } else {
    return Status(ONNXRUNTIME, FAIL, "Error: the graph is not acyclic.");
  }
}

bool FullyDefinedType(const TypeProto& type_proto) {
  switch (type_proto.value_case()) {
    case TypeProto::kTensorType: {
      auto& tensor_type = type_proto.tensor_type();
      return tensor_type.has_elem_type() && (tensor_type.elem_type() != TensorProto::UNDEFINED);
    }
    case TypeProto::kSparseTensorType: {
      auto& tensor_type = type_proto.sparse_tensor_type();
      return tensor_type.has_elem_type() && (tensor_type.elem_type() != TensorProto::UNDEFINED);
    }
    case TypeProto::kSequenceType: {
      auto& seq_type = type_proto.sequence_type();
      return seq_type.has_elem_type() && FullyDefinedType(seq_type.elem_type());
    }
    case TypeProto::kMapType: {
      auto& map_type = type_proto.map_type();
      return map_type.has_key_type() &&
             (map_type.key_type() != TensorProto::UNDEFINED) &&
             map_type.has_value_type() &&
             FullyDefinedType(map_type.value_type());
    }
    case TypeProto::kOpaqueType:
      return true;
    case TypeProto::VALUE_NOT_SET:
    default:
      return false;
  }
}

// function to handle type/shape inferencing of a subgraph.
// parameters are the Graph instance for the subgraph, the input types from the control flow node that contains
// the subgraph, and the vector to write the output from the inferencing.
using SubgraphInferencingFunc =
    std::function<Status(const Node&, Graph&, const std::vector<const TypeProto*>&, std::vector<const TypeProto*>&)>;

class GraphInferencerImpl : public ONNX_NAMESPACE::GraphInferencer {
 public:
  GraphInferencerImpl(const Node& node, Graph& graph, SubgraphInferencingFunc& inferencing_func)
      : node_{node}, graph_{graph}, inferencing_func_{inferencing_func} {
  }

  // Perform inferencing on the graph contained in GraphInferencer.
  // Returns the graph output types post-inferencing.
  // We ignore input_data currently. Re-consider if InferenceContextImpl::getInputData gets implemented
  std::vector<const TypeProto*> doInferencing(const std::vector<const TypeProto*>& input_types,
                                              const std::vector<const TensorProto*>& /*input_data*/) override {
    std::vector<const TypeProto*> output_types;

    auto status = inferencing_func_(node_, graph_, input_types, output_types);

    if (status != Status::OK()) {
      fail_type_inference("Graph attribute inferencing failed: ", status.ErrorMessage());
    }

    return output_types;
  }

 private:
  const Node& node_;
  Graph& graph_;
  SubgraphInferencingFunc& inferencing_func_;
};

// An implementation of the InferenceContext interface required by operator-specific
// shape inference for onnxruntime graphs.
class InferenceContextImpl : public ONNX_NAMESPACE::InferenceContext {
  using AttributeGraphMap = std::unordered_map<std::string, Graph*>;

 public:
  InferenceContextImpl(Node& node,
                       const AttributeGraphMap* subgraphs = nullptr,
                       SubgraphInferencingFunc* subgraph_inferencing_func = nullptr) noexcept
      : node_(node),
        attr_to_subgraph_map_{subgraphs},
        subgraph_inferencing_func_{subgraph_inferencing_func} {
    node_output_types_.resize(node.OutputDefs().size());
  }

  void RunInferencing() {
    auto schema = node_.Op();
    if (nullptr != schema) {
      schema->GetTypeAndShapeInferenceFunction()(*this);
    }
  }

  const std::vector<TypeProto> InferredOutputTypes() const { return node_output_types_; }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto& attribute_value_map = node_.GetAttributes();
    auto iter = attribute_value_map.find(name);
    if (iter == attribute_value_map.end()) {
      return nullptr;
    } else {
      return &iter->second;
    }
  }

  size_t getNumInputs() const noexcept override {
    return node_.InputDefs().size();
  }

  const TypeProto* getInputType(size_t index) const override {
    auto p_node_arg = node_.InputDefs().at(index);
    if ((nullptr != p_node_arg) && p_node_arg->Exists()) {
      return p_node_arg->TypeAsProto();
      // auto p_type_proto = p_node_arg->TypeAsProto();
      //if ((p_type_proto != nullptr) && p_type_proto->has_tensor_type()) {
      //  return &p_type_proto->tensor_type();
      //}
    }
    return nullptr;
  }

  size_t getNumOutputs() const noexcept override {
    return node_output_types_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    return &node_output_types_[index];
  }

  const TensorProto* getInputData(size_t) const override {
    // TODO: this interface should be implemented with initializers
    // so that more accurate shape inference could be done.
    return nullptr;
  }

  GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override {
    GraphInferencer* graph_inferencer = nullptr;

    if (attr_to_subgraph_map_ && subgraph_inferencing_func_) {
      auto attr_to_subgraph = attr_to_subgraph_map_->find(attribute_name);
      if (attr_to_subgraph != attr_to_subgraph_map_->cend()) {
        auto inferencer = std::make_unique<GraphInferencerImpl>(node_, *attr_to_subgraph->second,
                                                                *subgraph_inferencing_func_);
        graph_inferencer = inferencer.get();
        graph_inferencers_.push_back(std::move(inferencer));
      } else {
        fail_type_inference("No Graph instance was found for attribute ",
                            attribute_name, " in node ", node_.Name());
      }
    }

    return graph_inferencer;
  }

 private:
  Node& node_;
  // node_output_types_ will be populated by the operator-specific shape inference.
  std::vector<TypeProto> node_output_types_;
  const AttributeGraphMap* attr_to_subgraph_map_;
  SubgraphInferencingFunc* subgraph_inferencing_func_;
  std::vector<std::unique_ptr<GraphInferencerImpl>> graph_inferencers_;
};

Status Graph::InferAndVerifySubgraphTypes(const Node& node, Graph& subgraph,
                                          const std::vector<const TypeProto*>& input_types,
                                          std::vector<const TypeProto*>& output_types) {
  auto status = Status::OK();

  output_types.clear();

  auto& subgraph_inputs = subgraph.GetInputs();
  auto num_subgraph_inputs = subgraph_inputs.size();

  if (num_subgraph_inputs != input_types.size()) {
    return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Size mismatch validating subgraph inputs. Got ",
                                   input_types.size(), " inputs but subgraph requires ", subgraph_inputs.size());
  }

  // apply type/shape info to the subgraph's inputs
  for (size_t i = 0; i < num_subgraph_inputs; ++i) {
    const auto& input_type = *input_types[i];
    const auto& subgraph_input = *subgraph_inputs[i];

    NodeArg* mutable_nodearg = subgraph.GetNodeArg(subgraph_input.Name());
    status = mutable_nodearg->UpdateTypeAndShape(input_type);
    if (!status.IsOK()) {
      return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node.Name(), " ", status.ErrorMessage());
    }
  }

  // Apply any current input type/shape information to the Nodes in the subgraph that are implicitly
  // consuming NodeArg's from this scope or higher.
  // The NodeArg's that implicit_input_defs point to would have any type/shape inferencing applied to them
  // by now. As the subgraph is referring to the outer scope NodeArg, we simply replace any information in
  // the subgraph with the details from the outer scope NodeArg.
  auto implicit_input_defs = node.GetDefinitions().implicit_input_defs;
  for (const auto* implicit_node_arg : implicit_input_defs) {
    auto subgraph_nodearg = subgraph.GetNodeArg(implicit_node_arg->Name());

    // the implicit input defs may be for a nested subgraph, so it won't necessarily match here.
    // if that is the case, we will update the type/shape information when we descend into the
    // nested subgraph later.
    if (!subgraph_nodearg)
      continue;

    status = subgraph_nodearg->UpdateTypeAndShape(*implicit_node_arg);
    if (!status.IsOK()) {
      return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node.Name(), " ", status.ErrorMessage());
    }

    // all values above us should have a type by now due to ONNX requirements.
    if (subgraph_nodearg->Type() == nullptr)
      return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph input missing type.");
  }

  // now that we have handled the input types, do the type/shape inferencing for the subgraph
  // to flow the type/shape info through it
  status = subgraph.PerformTypeAndShapeInferencing();
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  auto& subgraph_outputs = subgraph.GetOutputs();
  for (const auto* output : subgraph_outputs) {
    output_types.push_back(output->TypeAsProto());
  }

  return Status::OK();
}

// Implementation of type-inference and type-checking for a single node
GSL_SUPPRESS(f .23)  // spurious warning about inferred_type never being checked for null
Status Graph::InferAndVerifyTypeMatch(Node& node, const OpSchema& op) {
  auto& node_name = node.Name();

  // if we're building a graph we permit outer scope node args to have no type
  // as the 'real' Resolve at runtime will have type inferencing
  auto is_outer_scope_nodearg = [this](const std::string& name) {
    return outer_scope_node_arg_names_.find(name) != outer_scope_node_arg_names_.cend();
  };

  // <k> index used to navigate node->InputDefs().
  int k = 0;
  std::unordered_map<std::string, DataType> type_parameter_to_type_map;

  for (size_t i = 0; i < node.InputArgCount().size(); ++i) {
    // Number of inputs corresponding to the i-th argument.
    const int arg_count = node.InputArgCount()[i];
    // The i-th formal parameter definition.
    auto op_formal_parameter = op.inputs()[i];

    // Check all <arg_count> actual parameters (corresponding to the k-th input)
    // match the formal parameter definition (i-th argument).
    for (int j = 0; j < arg_count; ++j, ++k) {
      auto& input_def = node.MutableDefinitions().input_defs[k];
      if (!input_def->Exists())
        continue;

      if (input_def->Type() == nullptr) {
        // if we are building a subgraph that uses outer scope values,
        // allow an empty type as it will be copied from the outer scope graph at runtime
        if (is_outer_scope_nodearg(input_def->Name()))
          continue;

        // Logic error: This should not happen if we properly checked that every use has
        // a corresponding def, for which type-inference already produced a valid type
        Status status(ONNXRUNTIME, FAIL,
                      "Node (" + node_name + ") input arg (" +
                          input_def->Name() + ") does not have type information set by parent node.");
        return status;
      }

      // Verify that the actual parameter's type is one of permitted types of the formal parameter
      DataType input_type = input_def->Type();
      auto& permitted_types = op_formal_parameter.GetTypes();
      if (0 == permitted_types.count(input_type)) {
        std::string null_pointer("(null)");
        if (input_type == nullptr) input_type = &null_pointer;
        // Type error in input model/graph.

        Status status(ONNXRUNTIME, INVALID_GRAPH,
                      "Type Error: Type '" + *input_type + "' of input parameter (" + input_def->Name() +
                          ") of operator (" + op.Name() + ") in node (" + node_name + ") is invalid.");
        return status;
      }

      // Check that type-parameters are bound to the same value:
      auto param_to_type_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
      if (type_parameter_to_type_map.end() == param_to_type_iter) {
        // Bind the corresponding type-parameter's value to the actual type:
        type_parameter_to_type_map[op_formal_parameter.GetTypeStr()] = input_type;
      } else if (param_to_type_iter->second != input_type) {
        // Type error in input model/graph:
        // The type-parameter T is bound to different values for different inputs.
        // E.g., Add(A,B) where A is of type "tensor(int32)" and B is of type "tensor(float)".
        // NOTE: for variadic arguments, this verification rule is currently applicable:
        // e.g., Concat/Max/Mean/Min/Sum all require all input tensors to be of same type.
        // However, this will need to be extended to handle the If-Then-Else and Loop
        // constructs in future which will have variadic inputs and outputs of different types.

        Status status(ONNXRUNTIME, FAIL,
                      "Type Error: Type parameter (" + op_formal_parameter.GetTypeStr() +
                          ") bound to different types (" + *(param_to_type_iter->second) +
                          " and " + *(input_def->Type()) +
                          " in node (" + node_name + ").");
        return status;
      }
    }
  }

  // Apply ONNX's type/shape inference to this node.
  // This will call InferAndVerifySubgraphTypes if the ONNX level type/shape inferencing for the Node attempts
  // to do subgraph type/shape inferencing (Scan/If/Loop nodes).
  // InferAndVerifySubgraphTypes will call PerformTypeAndShapeInferencing for the subgraph, which will recursively
  // handle type/shape inferencing for it.
  // Once that completes, the outputs from the node containing the subgraph will be updated, and the final values
  // returned here.
  SubgraphInferencingFunc func(Graph::InferAndVerifySubgraphTypes);
  auto node_subgraphs = subgraph_map_.find(node.Index());
  auto* subgraphs = node_subgraphs != subgraph_map_.cend() ? &node_subgraphs->second : nullptr;
  InferenceContextImpl context(node, subgraphs, &func);

  try {
    context.RunInferencing();
  } catch (const std::exception& ex) {
    return Status(ONNXRUNTIME, FAIL, ex.what());
  }

  const auto& onnx_inferred_types{context.InferredOutputTypes()};

  // Infer and verify node output arg type information.
  int i = -1;
  for (auto& output_def : node.MutableDefinitions().output_defs) {
    ++i;
    if (!output_def->Exists()) continue;

    // if the number of actual parameters exceeds the number of formal parameters,
    // then the op has variadic outputs and the trailing extra actual parameters
    // correspond to the last formal parameter. (The ONNX schema verification check
    // would have checked that the corresponding formal parameter is variadic.)

    const int num_formal_params = gsl::narrow_cast<int>(op.outputs().size());
    auto operand_index = std::min(i, num_formal_params - 1);
    auto op_formal_parameter = op.outputs().at(operand_index);

    const TypeProto& onnx_inferred_type = onnx_inferred_types[i];
    DataType existing_type = output_def->Type();
    DataType inferred_type = nullptr;

    // Infer output arg type if it is constrained to be of the same type as some input:
    // For example, the output of "Abs" is of the same type as its input.
    auto input_types_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
    if (type_parameter_to_type_map.end() != input_types_iter) {
      inferred_type = input_types_iter->second;
    } else if (1 == op_formal_parameter.GetTypes().size()) {
      // Infer output arg type if operator definition specifies unique output type:
      inferred_type = *(op_formal_parameter.GetTypes().begin());
    } else if (FullyDefinedType(onnx_inferred_type)) {
      // Use output type inferred by ONNX inference
      inferred_type = DataTypeUtils::ToType(onnx_inferred_type);
    } else if (existing_type != nullptr) {
      inferred_type = existing_type;
    } else {
      // This should not happen: indicates incompleteness in ONNX inference.
      Status status(ONNXRUNTIME, FAIL,
                    "Node (" + node_name + ") output arg (" + output_def->Name() + ") type inference failed");
      return status;
    }

    if ((existing_type != inferred_type) && (existing_type != nullptr)) {
      // A type exists for this output but does not match the inferred type.
      return Status(ONNXRUNTIME, FAIL,
                    "Type Error: Type (" + *existing_type + ") of output arg (" +
                        output_def->Name() + ") of node (" + node_name +
                        ") does not match expected type (" + *inferred_type + ").");
    }

    if (existing_type == nullptr)
      output_def->SetType(inferred_type);

    // Update output-shape if it was inferred:
    if (onnx_inferred_type.has_tensor_type()) {
      auto& tensor_type = onnx_inferred_type.tensor_type();
      if (tensor_type.has_shape()) {
        if (output_def->Shape() == nullptr) {
          output_def->SetShape(tensor_type.shape());
        } else {
          // we need to merge the shapes as a subgraph may have placeholder dimensions to represent the rank
          // that have no values.
          TypeProto_Tensor merge_target;
          (*merge_target.mutable_shape()) = *output_def->Shape();
          auto status = MergeShapeInfo(output_def->Name(), tensor_type, merge_target);
          if (!status.IsOK()) {
            return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node_name, " ", status.ErrorMessage());
          }

          output_def->SetShape(merge_target.shape());
        }
      }
    }
  }

  return Status::OK();
}

// Apply type-inference and type-checking to all inputs and initializers:
common::Status Graph::TypeCheckInputsAndInitializers() {
  // Check that the type of every input is specified:
  for (auto* graph_input : GetInputs()) {
    if (nullptr == graph_input->Type()) {
      Status status(ONNXRUNTIME, FAIL, "Model input (" + graph_input->Name() + ") does not have type information.");
      return status;
    }
  }

  // Note: The ONNX spec requires every initializer to be included in the graph input,
  // but onnxruntime relaxes this requirement for various reasons.

  // Infer/check type and shape for all initializers from their values
  for (auto& initializer_pair : name_to_initial_tensor_) {
    const std::string& name = initializer_pair.first;
    auto* node_arg = GetNodeArg(name);
    // If node_arg is null, we ignore this as a potentially unused initializer here
    if (nullptr != node_arg) {
      const TensorProto* tensor_proto = initializer_pair.second;
      TypeProto tensor_type;
      tensor_type.mutable_tensor_type()->set_elem_type(tensor_proto->data_type());
      auto inferred_type = DataTypeUtils::ToType(tensor_type);
      auto existing_type = node_arg->Type();
      if (nullptr == existing_type)
        node_arg->SetType(inferred_type);
      else if (inferred_type != existing_type) {
        return Status(ONNXRUNTIME, FAIL,
                      "Type Error: Value of initializer " + name + " does not match its type.");
      }

      // Set shape accordingly.
      TensorShapeProto inferred_shape;
      for (auto dim : tensor_proto->dims()) {
        inferred_shape.add_dim()->set_dim_value(dim);
      }
      const TensorShapeProto* p_existing_shape = node_arg->Shape();
      if (nullptr == p_existing_shape)
        node_arg->SetShape(inferred_shape);
      else {
        if (p_existing_shape->dim_size() != tensor_proto->dims_size())
          return Status(ONNXRUNTIME, FAIL,
                        "Type Error: Shape of initializer " + name + " does not match its type.");
        for (int i = 0; i < p_existing_shape->dim_size(); ++i) {
          auto& d = p_existing_shape->dim(i);
          if (d.has_dim_value() && (d.dim_value() != tensor_proto->dims(i)))
            return Status(ONNXRUNTIME, FAIL,
                          "Type Error: Shape of initializer " + initializer_pair.first + " does not match its type.");
        }
      }
    }
  }
  return Status::OK();
}

Status Graph::VerifyNodeAndOpMatch() {
  CheckerContext ctx;
  ctx.set_ir_version(gsl::narrow_cast<int>(IrVersion()));
  ctx.set_opset_imports(DomainToVersionMap());
  ctx.set_schema_registry(schema_registry_.get());

  LexicalScopeContext lsc{resolve_context_.inputs_and_initializers};

  // technically we could add values from Node.GetDefinitions().implicit_input_defs on a per-node basis inside
  // the below loop so that we only check against the specific outer dependencies of the node.
  // doing that requires lots of copies of LexicalScopeContext.output_names to clear out the per-Node values
  // after each loop. instead add all the outer scope values upfront so we can just accumulate new inner scope values
  // during each loop iteration.
  lsc.output_names.insert(resolve_context_.outer_scope_node_args.cbegin(),
                          resolve_context_.outer_scope_node_args.cend());

  for (auto node_index : nodes_in_topological_order_) {
    // Node verification.
    auto& node = *GetNode(node_index);

    NodeProto node_proto;
    node.ToProto(node_proto);
    auto& node_name = node.Name();
    auto& domain = node.Domain();

    if (!node.Op()) {
      try {
        checker::check_node(node_proto, ctx, lsc);
      } catch (const std::exception& ex) {
        return Status(ONNXRUNTIME, INVALID_GRAPH, ex.what());
      }

      auto maxInclusiveVersion = DomainToVersionMap().find(domain)->second;
      node.op_ = schema_registry_->GetSchema(node.OpType(), maxInclusiveVersion, node.Domain());

      if (node.op_ && node.op_->Deprecated()) {
        node.op_ = nullptr;
      }

      if (!node.op_) {
        ONNX_NAMESPACE::FunctionBuilderRegistry& function_registry =
            FunctionBuilderRegistry::OnnxInstance();
        auto onnx_function_proto = function_registry.GetFunction(node.OpType(), maxInclusiveVersion, ONNX_DOMAIN);
        if (!onnx_function_proto) {
          return Status(ONNXRUNTIME, FAIL, "Fatal error: " + node.OpType() + " is not a registered function/op");
        }
        auto func_ptr = std::make_unique<onnxruntime::FunctionImpl>(*this, node.Index(), onnx_function_proto);
        function_container_->functions_.push_back(std::move(func_ptr));
        node.SetFunctionBody(*function_container_->functions_.back());
      }
    }

    ONNXRUNTIME_RETURN_IF_ERROR(node.UpdateInputArgCount());

    // currently an Op is required by ValidateVersion, so we use gsl::not_null to validate that.
    // This may change in the future to allow a null Op
    const gsl::not_null<const OpSchema*> p_op{node.Op()};

    // Attribute verification and fill node attribute with
    // default value defined in operator definition if needed.
    // Fill node attribute with default value specified in operator definition if any.
    auto node_attributes = node.GetAttributes();
    for (auto attr_def : p_op->attributes()) {
      auto node_attr_iter = node_attributes.find(attr_def.first);
      if (node_attributes.end() == node_attr_iter) {
        // The attribute was not specified in the node.
        if (!attr_def.second.required) {
          if (attr_def.second.default_value.has_name()) {
            // Set default value to the node attributes.
            node.AddAttribute(attr_def.first, attr_def.second.default_value);
          }
          // TODO: Handle optional attribute but no default value specified in op definition.
        } else {
          Status status(ONNXRUNTIME, FAIL,
                        "Node (" + node_name + ") attribute (" + attr_def.first +
                            ") is required but not specified.");
          return status;
        }
      }
    }

    NO_CHANGE_ON_SYNC_FLAG(ONNXRUNTIME_RETURN_IF_ERROR(InferAndVerifyTypeMatch(node, *p_op)));

    // Accumulate output names of the iterated Node
    for (auto& output_name : node_proto.output()) {
      lsc.output_names.insert(output_name);
    }
  }

  return Status::OK();
}

Graph* Graph::GetMutableSubgraph(const NodeIndex node_index, const std::string& attribute_name) {
  const Graph* subgraph = GetSubgraph(node_index, attribute_name);
  return const_cast<Graph*>(subgraph);
}

const Graph* Graph::GetSubgraph(const NodeIndex node_index, const std::string& attribute_name) const {
  Graph* subgraph = nullptr;

  auto entry = subgraph_map_.find(node_index);

  if (entry != subgraph_map_.cend()) {
    auto& name_to_subgraph_map = entry->second;
    auto subgraph_iter = name_to_subgraph_map.find(attribute_name);
    if (subgraph_iter != name_to_subgraph_map.cend()) {
      subgraph = subgraph_iter->second;
    }
  }

  return subgraph;
}

Status Graph::CreateSubgraphs() {
  Status status = Status::OK();

  // don't use NodesInTopologicalOrder as we want CreateSubgraphs to recursively create subgraphs with no
  // dependency on PerformTopologicalSortAndCheckIsAcyclic having been called previously
  // to populate NodesInTopologicalOrder
  for (auto& node : Nodes()) {
    auto node_index = node.Index();
    if (subgraph_map_.find(node_index) != subgraph_map_.cend()) {
      // if we have an existing entry we have processed this node previously.
      // as the subgraph is loaded from a static GraphProto we assume nothing in
      // it could have changed and there's no point re-creating it.
      continue;
    }

    // check attributes of all nodes looking for GraphProto attributes, and create
    // the Graph instance for the subgraph contained in the GraphProto.
    for (auto& attr : node.attributes_) {
      bool has_subgraph = attr.second.has_g();
      if (has_subgraph) {
        auto& attr_name = attr.first;
        auto entry = subgraph_map_.find(node_index);

        // make sure this is new. internal logic error if it is not so using ONNXRUNTIME_ENFORCE.
        if (entry != subgraph_map_.cend()) {
          const auto& existing_entries = entry->second;
          ONNXRUNTIME_ENFORCE(existing_entries.find(attr_name) == existing_entries.cend(),
                              "Entry exists in node ", node_index, " for attribute ", attr_name);
        }

        auto& graph_proto = *attr.second.mutable_g();

        // create instance. need to call private ctor so can't use make_unique
        GSL_SUPPRESS(r .11)
        std::unique_ptr<Graph> subgraph{new Graph(*this, graph_proto)};

        // Recursively create any further subgraphs
        status = subgraph->CreateSubgraphs();
        ONNXRUNTIME_RETURN_IF_ERROR(status);

        subgraph_map_[node_index][attr_name] = subgraph.get();
        subgraphs_.push_back(std::move(subgraph));
      }
    }
  }

  return Status::OK();
}

Status Graph::VerifyInputAndInitializerNames() {
  std::unordered_set<std::string>& inputs_and_initializers = resolve_context_.inputs_and_initializers;

  for (auto* input : GetInputs()) {
    auto result = inputs_and_initializers.insert(input->Name());
    if (!result.second) {
      Status status(ONNXRUNTIME, FAIL,
                    "Error: Duplicate definition-site for (" + input->Name() + ").");
      return status;
    }
  }

  for (auto& initializer_pair : name_to_initial_tensor_) {
    GSL_SUPPRESS(es .84)
    inputs_and_initializers.insert(initializer_pair.first);
    // Initializers are expected to be included in inputs (according to ONNX spec).
    // onnxruntime relaxes this constraint. No duplicate-name check here.
  }

  return Status::OK();
}

Status Graph::InitInputsInitializersOutputs() {
  resolve_context_.Clear();

  // clear the previous relationships, as we re-create them when resolving.
  // same applies to the implicit input defs as they are built from any subgraphs within this graph.
  for (auto& node : Nodes()) {
    node.MutableRelationships().Clear();
    node.MutableDefinitions().implicit_input_defs.clear();
  }

  // add the subgraph pointers to the resolve context.
  for (auto& nodeid_to_subgraphs : subgraph_map_) {
    resolve_context_.node_to_subgraphs_map[nodeid_to_subgraphs.first] = {};

    for (auto& attr_name_to_subgraph : nodeid_to_subgraphs.second) {
      resolve_context_.node_to_subgraphs_map[nodeid_to_subgraphs.first].push_back(attr_name_to_subgraph.second);
    }
  }

  ONNXRUNTIME_RETURN_IF_ERROR(SetGraphInputsOutputs());
  ONNXRUNTIME_RETURN_IF_ERROR(VerifyInputAndInitializerNames());
  ONNXRUNTIME_RETURN_IF_ERROR(VerifyNoDuplicateName());

  return Status::OK();
}

Status Graph::PerformTypeAndShapeInferencing() {
  ONNXRUNTIME_RETURN_IF_ERROR(TypeCheckInputsAndInitializers());

  // type/shape inferencing on the nodes is done recursively as we need subgraph outputs
  // to be applied to Node outputs for the node containing the subgraph.
  // Call path is
  // VerifyNodeAndOpMatch
  //   Iterates Nodes
  //     Runs ONNX type/shape inferencing for each Node
  //      - If it hits a node with a subgraph, InferenceContext::getGraphAttributeInferencer is called
  //        by the ONNX level type/shape inferencing, which updates the subgraph inputs using GraphInferencerImpl
  //      - GraphInferencerImpl::doInferencing calls PerformTypeShapeInferencing to execute type/shape inferencing
  //        for all nodes in the subgraph. This leads to recursively handling all subgraphs contained in the node.
  //      - once we finish processing the subgraph/s we apply resultant type/shape information to the outputs
  //        of the node that contains the subgraph.
  ONNXRUNTIME_RETURN_IF_ERROR(VerifyNodeAndOpMatch());

  return Status::OK();
}

Status Graph::ForThisAndAllSubgraphs(std::function<Status(Graph&)> func) {
  auto status = func(*this);
  ONNXRUNTIME_RETURN_IF_ERROR(status);

  for (auto& subgraph : subgraphs_) {
    status = func(*subgraph);
    ONNXRUNTIME_RETURN_IF_ERROR(status);
  }
  return status;
}

Status Graph::Resolve() {
    const NodeArg *n = GetNodeArg("ReLU39_Output_0");
    Status s = Resolve(false);
  const NodeArg *n2 = GetNodeArg("ReLU39_Output_0");
  return s;
}

Status Graph::Resolve(bool no_proto_sync_required) {
  if (parent_graph_) {
    // Resolve must start at the top level graph in-order to handle outer scope
    // connections correctly, so recurse up to that level to start
    auto status = parent_graph_->Resolve(no_proto_sync_required);
    return status;
  }

  bool subgraphs_need_resolve = std::any_of(subgraphs_.cbegin(), subgraphs_.cend(),
                                            [](const std::unique_ptr<Graph>& graph) {
                                              return graph->GraphResolveNeeded();
                                            });

  if (!GraphResolveNeeded() && !subgraphs_need_resolve) {
    return Status::OK();
  }

  // Create the Graph instances for the subgraph/s in any nodes containing GraphProto attributes (Scan/If/Loop).
  // Do this upfront so we can recurse into them when building connections and doing type/shape inferencing.
  // Recursively creates any nested subgraphs.
  ONNXRUNTIME_RETURN_IF_ERROR(CreateSubgraphs());

  // init all graph/subgraphs. non-recursive.
  auto init_func = [](Graph& graph) { return graph.InitInputsInitializersOutputs(); };
  ONNXRUNTIME_RETURN_IF_ERROR(ForThisAndAllSubgraphs(init_func));

  // recursively set the outer scope node args.
  ONNXRUNTIME_RETURN_IF_ERROR(SetOuterScopeNodeArgs(resolve_context_.outer_scope_node_args));

  std::vector<std::string> outer_scope_node_args_consumed;

  // recursively build connections between nodes in this graph and all subgraphs
  ONNXRUNTIME_RETURN_IF_ERROR(BuildConnections(outer_scope_node_args_consumed));
  ONNXRUNTIME_ENFORCE(outer_scope_node_args_consumed.empty(),
                      "Shouldn't be possible to have NodeArgs that haven't been handled already.");

  // topological sort of this and any subgraphs is non-recursive
  auto topo_sort_func = [](Graph& graph) { return graph.PerformTopologicalSortAndCheckIsAcyclic(); };
  ONNXRUNTIME_RETURN_IF_ERROR(ForThisAndAllSubgraphs(topo_sort_func));

  // type/shape validation and inferencing on this and any subgraphs
  // recurses into subgraphs via the ONNX checker, which descends into the GraphProto in node attributes
  // which define a subgraph.
  ONNXRUNTIME_RETURN_IF_ERROR(PerformTypeAndShapeInferencing());

  // perform the final steps for this graph and all subgraphs
  auto finalize_func = [&no_proto_sync_required](Graph& graph) {
            graph.CleanUnusedInitializers();
            graph.GraphResolveNeeded(false);

            // if we are resolving immediately after loading from a GraphProto, we don't need to
            // do a proto sync
            if (no_proto_sync_required) {
                graph.GraphProtoSyncNeeded(false);
            }

            return Status::OK(); };

  ONNXRUNTIME_RETURN_IF_ERROR(ForThisAndAllSubgraphs(finalize_func));

  return Status::OK();
}

const std::string& Graph::Name() const noexcept {
  return graph_proto_->name();
}

void Graph::SetName(const std::string& name) {
  graph_proto_->set_name(name);
}

const std::string& Graph::Description() const noexcept {
  return graph_proto_->doc_string();
}

void Graph::SetDescription(const std::string& description) {
  graph_proto_->set_doc_string(description);
}

void Graph::AddInitializedTensor(const TensorProto& tensor) {
  if (name_to_initial_tensor_.end() != name_to_initial_tensor_.find(tensor.name())) {
    return;
  }

  const gsl::not_null<TensorProto*> tensor_added{graph_proto_->add_initializer()};
  *(tensor_added) = tensor;
  name_to_initial_tensor_[tensor.name()] = tensor_added;

  if (!GraphLoadedFromModelFile(graph_proto_)) {
    // make sure there is a NodeArg for the initializer as SetGraphInputsOutputs will add it to the graph inputs
    TypeProto t;
    t.mutable_tensor_type()->set_elem_type(tensor.data_type());
    auto shape = t.mutable_tensor_type()->mutable_shape();
    for (auto dim : tensor.dims())
      shape->add_dim()->set_dim_value(dim);

    ONNXRUNTIME_IGNORE_RETURN_VALUE(GetOrCreateNodeArg(tensor.name(), &t));
  }

  SetGraphProtoSyncNeeded();
  SetGraphResolveNeeded();
}

void Graph::RemoveInitializedTensor(const std::string& tensor_name) {
  auto iter = name_to_initial_tensor_.find(tensor_name);
  if (name_to_initial_tensor_.end() != iter) {
    name_to_initial_tensor_.erase(tensor_name);
    SetGraphProtoSyncNeeded();
    SetGraphResolveNeeded();
  }
}

bool Graph::GetInitializedTensor(const std::string& tensor_name, const TensorProto*& value) const {
  auto iter = name_to_initial_tensor_.find(tensor_name);
  if (name_to_initial_tensor_.end() == iter) {
    value = nullptr;
    return false;
  }
  value = iter->second;
  return true;
}

void Graph::CleanAllInitializedTensors() noexcept {
  name_to_initial_tensor_.clear();
  removed_initializer_indexes_.clear();

  // Clearing RepeatedPtrFields does not free objects' memory. The memory is retained
  // and can be reused. Need to explicitly release the cleared objects and free the
  // memory.
  graph_proto_->mutable_initializer()->Clear();
  const int num_cleared = graph_proto_->initializer().ClearedCount();
  for (int i = 0; i < num_cleared; i++) {
    delete graph_proto_->mutable_initializer()->ReleaseCleared();
  }
}

const InitializedTensorSet& Graph::GetAllInitializedTensors() const noexcept {
  return name_to_initial_tensor_;
}

const std::vector<const NodeArg*>& Graph::GetValueInfo() const noexcept {
  return value_info_;
}

std::vector<NodeArg*> Graph::CreateNodeArgs(const google::protobuf::RepeatedPtrField<std::string>& names,
                                            const ArgNameToTypeMap& name_to_type_map) {
  const auto name_to_type_map_end = name_to_type_map.end();
  std::vector<NodeArg*> results;
  results.reserve(names.size());

  for (auto& name : names) {
    const TypeProto* type = nullptr;

    auto name_to_type_iter = name_to_type_map.find(name);
    if (name_to_type_iter != name_to_type_map_end) {
      // This node input arg type/shape does exist in graph proto.
      // Assign type/shape information to node input arg.
      type = &(name_to_type_iter->second);
    }

    auto node_arg = &GetOrCreateNodeArg(name, type);
    results.push_back(node_arg);
  }

  return results;
}

Node* Graph::AddNode(const Node& other) {
  const auto& definitions = other.GetDefinitions();

  auto new_node = AddNode(other.Name(), other.OpType(), other.Description(),
                          definitions.input_defs,
                          definitions.output_defs,
                          &other.GetAttributes(),
                          other.Domain());

  return new_node;
}

Node* Graph::AddNode(const NodeProto& node_proto,
                     const ArgNameToTypeMap& name_to_type_map) {
  auto input_defs = CreateNodeArgs(node_proto.input(), name_to_type_map);
  auto output_defs = CreateNodeArgs(node_proto.output(), name_to_type_map);

  const int num_attributes = node_proto.attribute_size();
  NodeAttributes attributes;
  attributes.reserve(num_attributes);

  for (int i = 0; i < num_attributes; ++i) {
    auto& attr = node_proto.attribute(i);
    attributes[attr.name()] = attr;
  }

  return AddNode(node_proto.name(),
                 node_proto.op_type(),
                 node_proto.doc_string(),
                 input_defs,
                 output_defs,
                 &attributes,
                 node_proto.domain());
}

std::string Graph::GenerateNodeArgName(const std::string& base_name) {
  std::string new_name;
  do {
    std::ostringstream str;
    str << base_name << "_" << name_generator_++;
    new_name = str.str();
  } while (node_args_.find(new_name) != node_args_.end());
  return new_name;
}

std::string Graph::GenerateNodeName(const std::string& base_name) {
  std::string new_name;
  bool keep_going = true;

  do {
    std::ostringstream str;
    str << base_name << "_" << name_generator_++;
    new_name = str.str();

    keep_going = std::find_if(nodes_.cbegin(), nodes_.cend(), [&new_name](const std::unique_ptr<Node>& n) {
                   return (n != nullptr) && (n->Name() == new_name);
                 }) != nodes_.end();
  } while (keep_going);

  return new_name;
}

Node* Graph::AddNode(const std::string& name,
                     const std::string& op_type,
                     const std::string& description,
                     const std::vector<NodeArg*>& input_args,
                     const std::vector<NodeArg*>& output_args,
                     const NodeAttributes* attributes,
                     const std::string& domain) {
  std::vector<NodeArg*> inputs, outputs;
  inputs.resize(input_args.size());
  outputs.resize(output_args.size());
  int i = 0;
  for (auto input_arg : input_args) {
    inputs[i++] = &GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
  }
  i = 0;
  for (auto output_arg : output_args) {
    outputs[i++] = &GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
  }

  const gsl::not_null<Node*> node = AllocateNode();
  node->Init(name, op_type, description, inputs, outputs, attributes, domain);
  if (0 != op_type.compare(kNoOp)) {
    graph_proto_sync_needed_ = true;
  }

  return node;
}

bool Graph::RemoveNode(NodeIndex p_index) {
  return ReleaseNode(p_index);
}

bool Graph::AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index) {
  if (nodes_.size() <= src_node_index ||
      nodes_.size() <= dst_node_index ||
      nullptr == nodes_[src_node_index] ||
      nullptr == nodes_[dst_node_index]) {
    // Invalid node indexes specified.
    return false;
  }

  GSL_SUPPRESS(es .84) {  // ignoring return from insert()
    nodes_[src_node_index]->MutableRelationships().output_edges.insert(Node::EdgeEnd(*nodes_[dst_node_index]));
    nodes_[dst_node_index]->MutableRelationships().input_edges.insert(Node::EdgeEnd(*nodes_[src_node_index]));
    nodes_[dst_node_index]->MutableRelationships().control_inputs.insert(nodes_[src_node_index]->Name());
  }

  return true;
}

const GraphProto& Graph::ToGraphProto() {
  if (!GraphProtoSyncNeeded()) {
    return *graph_proto_;
  }

  // Nodes.
  graph_proto_->clear_node();
  GraphViewer graph_viewer(*this);
  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    const gsl::not_null<NodeProto*> node_proto{graph_proto_->add_node()};
    const gsl::not_null<Node*> p_node{GetNode(node_idx)};
    p_node->ToProto(*node_proto);
  }

  if (!removed_initializer_indexes_.empty()) {
    // Move initializers.
    std::sort(removed_initializer_indexes_.begin(), removed_initializer_indexes_.end());
    int lastInUseInitializerIndex = graph_proto_->initializer_size() - 1;
    int start = 0, end = gsl::narrow_cast<int>(removed_initializer_indexes_.size()) - 1;
    int lastRemovedInitializerIndex = removed_initializer_indexes_[end];

    for (; start <= end; start++) {
      // Find a lastInUseInitializer.
      while (start <= end && lastInUseInitializerIndex == lastRemovedInitializerIndex) {
        graph_proto_->mutable_initializer()->RemoveLast();
        lastInUseInitializerIndex--;
        end--;
        if (start <= end) {
          lastRemovedInitializerIndex = removed_initializer_indexes_[end];
        }
      }

      if (start <= end) {
        // Copy the <lastInUseInitializerIndex> initializer in use to the <start> slot which is removed.
        *graph_proto_->mutable_initializer(removed_initializer_indexes_[start]) = graph_proto_->initializer(lastInUseInitializerIndex);
        graph_proto_->mutable_initializer()->RemoveLast();
        lastInUseInitializerIndex--;
      }
    }
    removed_initializer_indexes_.clear();
  }

  // Sync graph inputs/outputs/valueInfo.
  SyncGraphInputsOutputs();

  GraphProtoSyncNeeded(false);

  return *graph_proto_;
}

void Graph::SyncGraphInputsOutputs() {
  graph_proto_->clear_input();
  graph_proto_->clear_output();
  graph_proto_->clear_value_info();

  for (const auto* input_arg : GetInputsIncludingInitializers()) {
    *(graph_proto_->mutable_input()->Add()) = input_arg->ToProto();
  }

  for (const auto* output_arg : GetOutputs()) {
    *(graph_proto_->mutable_output()->Add()) = output_arg->ToProto();
  }

  for (const auto* value_info : value_info_) {
    *(graph_proto_->mutable_value_info()->Add()) = value_info->ToProto();
  }
}

void Graph::CleanUnusedInitializers() {
  std::unordered_set<std::string> used_args;

  const auto& inputs = GetInputs();
  const auto& outputs = GetOutputs();

  std::for_each(inputs.cbegin(), inputs.cend(), [&used_args](const NodeArg* input) {
    ONNXRUNTIME_IGNORE_RETURN_VALUE(used_args.insert(input->Name()));
  });

  std::for_each(outputs.cbegin(), outputs.cend(), [&used_args](const NodeArg* output) {
    ONNXRUNTIME_IGNORE_RETURN_VALUE(used_args.insert(output->Name()));
  });

  for (const auto& node : Nodes()) {
    node.ForEachInputDef([&used_args](const onnxruntime::NodeArg* def) {
      ONNXRUNTIME_IGNORE_RETURN_VALUE(used_args.insert(def->Name()));
    });
  }

  std::vector<std::string> erase_list;
  auto end = used_args.end();
  for (const auto& pv : name_to_initial_tensor_) {
    const std::string& name = pv.first;
    if (used_args.find(name) == end) {
      LOGS_DEFAULT(WARNING) << name << " exists in this graph's initializers but it is not used by any node";
      erase_list.push_back(name);
    }
  }

  std::for_each(erase_list.cbegin(), erase_list.cend(),
                [this](const std::string& name) { name_to_initial_tensor_.erase(name); });
}

GSL_SUPPRESS(es .84)  // warning about ignoring return value from insert(...)
Status Graph::SetGraphInputsOutputs() {
  // Reset graph inputs/outputs/value info state.
  graph_inputs_excluding_initializers_.clear();
  graph_inputs_including_initializers_.clear();
  graph_outputs_.clear();
  value_info_.clear();

  // Flag indicates that this graph is loaded from model file.
  // If it's true, then graph inputs and outputs will keep the same
  // as what are specified in the model, otherwise, graph inputs
  // and outputs will be inferred.
  const bool loaded_from_model_file = GraphLoadedFromModelFile(graph_proto_);

  // if something is coming from outer scope, consider it already added
  std::unordered_set<std::string> added_input_names{outer_scope_node_arg_names_};

  if (loaded_from_model_file) {
    // Collect all graph inputs/outputs specified in original graph proto
    std::unordered_set<std::string> specified_graph_inputs;
    std::unordered_set<std::string> specified_graph_outputs;
    std::unordered_set<std::string> specified_graph_value_info;
    std::unordered_set<std::string> specified_initializers;
    std::unordered_map<std::string, const NodeArg*> input_name_to_node_arg;
    std::unordered_map<std::string, const NodeArg*> output_name_to_node_arg;

    for (auto& graph_output : graph_proto_->output()) {
      specified_graph_outputs.insert(graph_output.name());
    }

    for (auto& graph_value_info : graph_proto_->value_info()) {
      specified_graph_value_info.insert(graph_value_info.name());
    }

    for (auto& initializer : graph_proto_->initializer()) {
      specified_initializers.insert(initializer.name());
    }

    for (auto& graph_input : graph_proto_->input()) {
      // add all graph inputs to input_name_to_node_arg
      auto& name = graph_input.name();
      const auto* node_arg = GetNodeArg(name);
      ONNXRUNTIME_ENFORCE(node_arg, "Graph ctor should have created NodeArg for initializer.");
      input_name_to_node_arg.insert({name, node_arg});

      // only add non-initializer to specified_graph_inputs
      if (specified_initializers.find(name) == specified_initializers.end())
        specified_graph_inputs.insert(name);
    }

    // add non-initializer outputs
    for (const auto& node : Nodes()) {
      for (const auto* output_def : node.OutputDefs()) {
        ONNXRUNTIME_IGNORE_RETURN_VALUE(specified_graph_outputs.erase(output_def->Name()));
        output_name_to_node_arg.insert({output_def->Name(), output_def});
      }
    }

    // add any outputs using initializer
    if (specified_graph_outputs.size() > 0) {
      for (const auto& name : specified_initializers) {
        ONNXRUNTIME_IGNORE_RETURN_VALUE(specified_graph_outputs.erase(name));
        output_name_to_node_arg.insert({name, GetNodeArg(name)});
      }
    }

    if (!specified_graph_outputs.empty()) {
      std::string missing_list;
      for (auto& name : specified_graph_outputs)
        missing_list += name + " ";
      return Status(ONNXRUNTIME, FAIL, "Some graph outputs do not exist in the graph. (" + missing_list + ")");
    }

    for (const auto& node : Nodes()) {
      // Go thru all node's inputs.
      for (const auto* input_arg : node.InputDefs()) {
        if (!input_arg->Exists()) {
          // It's an optional input and does not exist in this case.
          continue;
        }

        if (specified_graph_inputs.end() != specified_graph_inputs.find(input_arg->Name())) {
          if (added_input_names.insert(input_arg->Name()).second) {
            // The node input is specified as graph input.
            input_name_to_node_arg.insert({input_arg->Name(), input_arg});
          }
          continue;
        }

        auto output_arg_iter = output_name_to_node_arg.find(input_arg->Name());
        if (output_name_to_node_arg.end() == output_arg_iter &&
            specified_initializers.end() == specified_initializers.find(input_arg->Name())) {
          // The node input is not specified as graph input,
          // and it's not fed by another node neither.
          if (!IsSubgraph()) {
            return Status(ONNXRUNTIME, FAIL, "Node input (" + input_arg->Name() + ") should be a graph input or initializer.");
          }

          // TODO: Do we need to do a comprehensive check that the input is coming from the outer scope or is it
          // fine to catch this issue later?
        }

        if (specified_graph_value_info.erase(input_arg->Name()) >= 1) {
          value_info_.push_back(input_arg);
        }
      }
    }

    // preserve input order
    for (auto& graph_input : graph_proto_->input()) {
      auto& name = graph_input.name();
      auto node_arg_iter = input_name_to_node_arg.find(name);
      ONNXRUNTIME_ENFORCE(node_arg_iter != input_name_to_node_arg.cend(),
                          "All inputs and initializers should have entries. Missing ", name);

      graph_inputs_including_initializers_.push_back(node_arg_iter->second);

      if (specified_initializers.find(name) == specified_initializers.end()) {
        graph_inputs_excluding_initializers_.push_back(node_arg_iter->second);
      }
    }

    // preserve output order
    for (auto& graph_output : graph_proto_->output()) {
      graph_outputs_.push_back(output_name_to_node_arg.at(graph_output.name()));
    }
  } else {
    std::unordered_map<std::string, const NodeArg*> output_name_to_node_arg;
    std::vector<std::string> ordered_output_names;

    // add any explicitly ordered inputs
    for (auto* node_arg : graph_input_order_) {
      if (!node_arg || !node_arg->Exists()) {
        return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid entry in explicitly ordered inputs");
      }

      added_input_names.insert(node_arg->Name());
      graph_inputs_including_initializers_.push_back(node_arg);
      if (name_to_initial_tensor_.find(node_arg->Name()) == name_to_initial_tensor_.end()) {
        graph_inputs_excluding_initializers_.push_back(node_arg);
      }
    }

    // add any explicitly ordered outputs
    for (auto* node_arg : graph_output_order_) {
      if (!node_arg || !node_arg->Exists()) {
        return ONNXRUNTIME_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid entry in explicitly ordered outputs");
      }
      output_name_to_node_arg.insert({node_arg->Name(), node_arg});
      ordered_output_names.push_back(node_arg->Name());
    }

    // add all other outputs
    for (const auto& node : Nodes()) {
      for (const auto* output_def : node.OutputDefs()) {
        if (output_def->Exists()) {
          auto& name = output_def->Name();
          // check it wasn't in the explicitly ordered outputs
          if (output_name_to_node_arg.find(name) == output_name_to_node_arg.cend()) {
            output_name_to_node_arg.insert({name, output_def});
            ordered_output_names.push_back(name);
          }
        }
      }
    }

    // Init graph output args with copy of all node output args.
    auto graph_output_args = output_name_to_node_arg;
    std::unordered_set<Node*> inner_nodes;

    for (const auto& node : Nodes()) {
      // Go thru all node's inputs.
      for (const auto* input_arg : node.InputDefs()) {
        if (!input_arg->Exists()) {
          // It's an optional input and does not exist in this case.
          continue;
        }

        auto output_arg_iter = output_name_to_node_arg.find(input_arg->Name());
        if (output_name_to_node_arg.end() == output_arg_iter) {
          // This input arg should be fed when running evaluation.
          // it should be a graph input.
          const std::string& name = input_arg->Name();
          if (added_input_names.end() == added_input_names.find(name)) {
            // This graph input has not been added into <graph_inputs_>.
            graph_inputs_including_initializers_.push_back(input_arg);

            if (name_to_initial_tensor_.find(name) == name_to_initial_tensor_.end()) {
              graph_inputs_excluding_initializers_.push_back(input_arg);
            }

            added_input_names.insert(input_arg->Name());
          }
        } else if (graph_output_args.erase(output_arg_iter->first) >= 1) {
          // Remove the output arg name from graph outputs since it's
          // the input of this node, which we call it intermediate result
          // and store it in <m_valueinfo>.
          value_info_.push_back(input_arg);
        }
      }
    }

    // Make sure all initializers appear as graph inputs as per ONNX requirements
    for (auto i : name_to_initial_tensor_) {
      if (added_input_names.find(i.first) == added_input_names.cend()) {
        auto* na = GetNodeArg(i.first);
        graph_inputs_including_initializers_.push_back(na);
      }
    }

    // Set graph outputs
    auto end = graph_output_args.end();
    for (auto& name : ordered_output_names) {
      auto graph_output = graph_output_args.find(name);
      if (graph_output != end) {
        graph_outputs_.push_back(graph_output->second);
      }
    }
  }

  return Status::OK();
}

// calling private ctor
GSL_SUPPRESS(r .11)
gsl::not_null<Node*> Graph::AllocateNode() {
  std::unique_ptr<Node> new_node(new Node(nodes_.size(), *this));
  Node* node{new_node.get()};

  nodes_.push_back(std::move(new_node));
  ++num_of_nodes_;
  graph_resolve_needed_ = true;

  return gsl::not_null<Node*>{node};
}

// TODO: Does this need (and maybe AllocateNode) to be threadsafe so nodes_ and num_of_nodes_ managed more carefully?
bool Graph::ReleaseNode(NodeIndex index) {
  if (index >= nodes_.size()) {
    return false;
  }

  // index is valid, but the entry may already be empty
  if (nodes_[index] != nullptr) {
    nodes_[index] = nullptr;
    --num_of_nodes_;
    graph_proto_sync_needed_ = true;
    graph_resolve_needed_ = true;
  }

  return true;
}

IOnnxRuntimeOpSchemaCollectionPtr Graph::GetSchemaRegistry() const {
  return schema_registry_;
}

Node* Graph::FuseSubGraph(std::unique_ptr<::onnxruntime::IndexedSubGraph> sub_graph, const std::string& fused_node_name) {
  ONNXRUNTIME_ENFORCE(nullptr != sub_graph && nullptr != sub_graph->GetMetaDef());

  auto func_meta_def = sub_graph->GetMetaDef();
  ONNXRUNTIME_ENFORCE(nullptr != func_meta_def);
  std::vector<NodeArg*> input_args, output_args;
  for (auto& arg_name : func_meta_def->inputs) {
    input_args.push_back(GetNodeArg(arg_name));
  }
  for (auto& arg_name : func_meta_def->outputs) {
    output_args.push_back(GetNodeArg(arg_name));
  }
  auto fused_node = AddNode(fused_node_name,
                            func_meta_def->name,
                            func_meta_def->doc_string,
                            input_args,
                            output_args,
                            &func_meta_def->attributes,
                            func_meta_def->domain);

  fused_node->SetNodeType(Node::Type::Fused);
  function_container_->functions_.push_back(MakeFunction(*this, std::move(sub_graph)));
  fused_node->SetFunctionBody(*(function_container_->functions_.back().get()));

  // Remove nodes fused above.
  auto& sub_graph_ref = function_container_->functions_.back()->GetIndexedSubGraph();
  for (auto node_index : sub_graph_ref.nodes) {
    RemoveNode(node_index);
  }
  return fused_node;
}

Graph::~Graph() {
  // nothing to do, but we put it here so we don't need to fully define types in Graph that are held in unique_ptr
  // such as   std::unique_ptr<FunctionContainer> function_container_;
}
}  // namespace onnxruntime
