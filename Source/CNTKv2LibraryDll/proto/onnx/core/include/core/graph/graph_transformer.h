// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/rewrite_rule.h"

namespace onnxruntime {

// A graph transformer interface. A graph transformer transforms a graph in-place.
class GraphTransformer {
 public:
  GraphTransformer(const std::string& name, const std::string& desc)
      : name_(name), desc_(desc) {
  }

  virtual ~GraphTransformer() = default;

  // The name of this graph transformer.
  const std::string& Name() const noexcept {
    return name_;
  }

  // An description of this graph transformer.
  const std::string& Description() const noexcept {
    return desc_;
  }

  // Apply <*this> transformation to a specific graph.
  // Transformation happens in place.
  // The return value of "modified" indicates if the graph was modified or not.
  virtual ::onnxruntime::common::Status Apply(Graph& graph, bool& modified) const = 0;

 private:
  LOTUS_DISALLOW_COPY_ASSIGN_AND_MOVE(GraphTransformer);

  const std::string name_;
  const std::string desc_;
};

// Rule based graph transformer.
// It provides API to register rewrite rules, and API to apply for
// all applicable rules against one graph.

// Represents a IGraphTransformer determined by a set of rewrite-rules.
// The transformer will apply all the rewrite-rules iteratively as
// determined by the underlying rewriting-strategy.
// TODO: Several rewriting-strategies are possible, with different tradeoffs.
// To begin with, we may use a simple, bottom-up, rewriting strategy.
class RuleBasedGraphTransformer : public GraphTransformer {
 public:
  // Register a rewriting rule.
  // TODO (revisit needed): Using OpSignature* here will ask that OpSignature
  // should be stored globally. Otherwise, there will be multiple addresses/pointers
  // for the same operator or function. To avoid this, we may use OpSignature ID
  // as the key, which should be name_domain_version.
  ::onnxruntime::common::Status Register(const ONNX_NAMESPACE::OpSchema* op, std::unique_ptr<RewriteRule> rule) {
    op_to_rules_[op].push_back(std::move(rule));
    return ::onnxruntime::common::Status::OK();
  }

  // Apply for all applicable rules against one graph.
  ::onnxruntime::common::Status Apply(Graph&, bool&) const override {
    LOTUS_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }

 private:
  using RewriteRuleSet = std::unordered_map<const ONNX_NAMESPACE::OpSchema*, std::vector<std::unique_ptr<RewriteRule>>>;

  RewriteRuleSet op_to_rules_;
};
}  // namespace onnxruntime
