#if 0
#pragma once

#include "marian.h"
//#include "layers/factory.h"
//#include "layers/generic.h"

namespace marian {
namespace mlp {

struct LayerFactory : public Factory {
  LayerFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}
  LayerFactory(const LayerFactory&) = default;
  LayerFactory(LayerFactory&&) = default;

  virtual ~LayerFactory() {}

  template <typename Cast>
  inline Ptr<Cast> as() {
    return std::dynamic_pointer_cast<Cast>(shared_from_this());
  }

  template <typename Cast>
  inline bool is() {
    return as<Cast>() != nullptr;
  }

  virtual Ptr<Layer> construct() = 0;
};

class DenseFactory : public LayerFactory {
protected:
  std::vector<std::pair<std::string, std::string>> tiedParams_;
  std::vector<std::pair<std::string, std::string>> tiedParamsTransposed_;

public:
  DenseFactory(Ptr<ExpressionGraph> graph) : LayerFactory(graph) {}

  Accumulator<DenseFactory> tie(const std::string& param,
                                const std::string& tied) {
    tiedParams_.push_back({param, tied});
    return Accumulator<DenseFactory>(*this);
  }

  Accumulator<DenseFactory> tie_transposed(const std::string& param,
                                           const std::string& tied) {
    tiedParamsTransposed_.push_back({param, tied});
    return Accumulator<DenseFactory>(*this);
  }

  Ptr<Layer> construct() {
    auto dense = New<Dense>(graph_, options_);
    for(auto& p : tiedParams_)
      dense->tie(p.first, p.second);
    for(auto& p : tiedParamsTransposed_)
      dense->tie_transposed(p.first, p.second);
    return dense;
  }
};

typedef Accumulator<DenseFactory> dense;

class MLP {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

  std::vector<Ptr<Layer>> layers_;

public:
  MLP(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : graph_(graph), options_(options) {}

  template <typename... Args>
  Expr apply(Args... args) {
    std::vector<Expr> av = {args...};

    Expr output;
    if(av.size() == 1)
      output = layers_[0]->apply(av[0]);
    else
      output = layers_[0]->apply(av);

    for(int i = 1; i < layers_.size(); ++i)
      output = layers_[i]->apply(output);

    return output;
  }

  void push_back(Ptr<Layer> layer) { layers_.push_back(layer); }
};

class MLPFactory : public Factory {
private:
  std::vector<Ptr<LayerFactory>> layers_;

public:
  MLPFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  Ptr<MLP> construct() {
    auto mlp = New<MLP>(graph_, options_);
    for(auto layer : layers_) {
      layer->getOptions()->merge(options_);
      mlp->push_back(layer->construct());
    }
    return mlp;
  }

  Ptr<MLP> operator->() { return construct(); }

  template <class LF>
  Accumulator<MLPFactory> push_back(const LF& lf) {
    layers_.push_back(New<LF>(lf));
    return Accumulator<MLPFactory>(*this);
  }
};

typedef Accumulator<MLPFactory> mlp;
}
}
#endif
