#pragma once

#include <string>

//#include "data/batch.h"
//#include "graph/expression_graph.h"

namespace marian {
namespace models {

class ModelBase {
public:
  virtual void load(Ptr<ExpressionGraph>, const std::string&) = 0;

  virtual void save(Ptr<ExpressionGraph>,
                    const std::string&,
                    bool saveTranslatorConfig = false)
      = 0;

  virtual Expr build(Ptr<ExpressionGraph> graph,
                     Ptr<data::Batch> batch,
                     bool clearGraph = true)
      = 0;

  //virtual Ptr<data::BatchStats> collectStats(Ptr<ExpressionGraph> graph,
  //                                           size_t multiplier = 1)
  //    = 0;

  virtual void clear(Ptr<ExpressionGraph> graph) = 0;
};
}
}
