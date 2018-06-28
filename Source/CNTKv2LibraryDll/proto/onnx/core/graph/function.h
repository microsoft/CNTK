#pragma once

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/rewrite_rule.h"

namespace LotusIR {
// TODO: Design a loose way to register rewrite rules into RuleBasedGraphTransformer.
// Function representation class.
class Function : public GraphBase {
 public:
  // Get <*this> function's schema.
  const OpSchema& GetSchema() const noexcept {
    return schema_;
  }

 private:
  OpSchema schema_;
};

// A function-inlining rewrite-rule. The plan with ONNX is to capture most optimizations
// as function-inlining or function-extraction.
class FunctionInliner : public RewriteRule {
 public:
  FunctionInliner(const std::string& name, const std::string& desc,
                  const Function& function)
    : RewriteRule(name, desc) {
    (function);
  }

  Status Apply(GraphEditor graph_editor, Node* node, bool* modified) override {
    (graph_editor);
    (node);
    (modified);
    LOTUS_NOT_IMPLEMENTED;
    return Status::OK();
  }
};

// A function-extraction rewrite-rule is the dual of function-inlining.
// It identifies occurrences of the body of a function-definition and
// replaces it by a call to the function.
class FunctionExtraction : public RewriteRule {
 public:
  FunctionExtraction(const std::string& name, const std::string& desc,
                     const Function& function)
    : RewriteRule(name, desc) {
    (function);
  }

  Status Apply(GraphEditor graph_editor, Node* node, bool* modified) override {
    (graph_editor);
    (node);
    (modified);
    LOTUS_NOT_IMPLEMENTED;
    return Status::OK();
  }
};

}  // namespace LotusIR
