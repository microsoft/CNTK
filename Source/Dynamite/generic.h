#pragma once

//#include "common/definitions.h"
//#include "common/options.h"
//#include "graph/expression_graph.h"
//#include "graph/expression_operators.h"
//#include "layers/factory.h"
//#include "layers/param_initializers.h"

namespace marian {
namespace mlp {
enum struct act : int { linear, tanh, logit, ReLU, LeakyReLU, PReLU, swish };
}
}

//YAML_REGISTER_TYPE(marian::mlp::act, int)

namespace marian {
namespace mlp {

class Layer {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

public:
  Layer(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : graph_(graph), options_(options) {}

  template <typename T>
  T opt(const std::string key) {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string key, T defaultValue) {
    return options_->get<T>(key, defaultValue);
  }

  virtual Expr apply(const std::vector<Expr>&) = 0;
  virtual Expr apply(Expr) = 0;
};

class Dense : public Layer {
private:
  std::vector<Expr> params_;
  std::map<std::string, Expr> tiedParams_;

public:
  Dense(Ptr<ExpressionGraph> graph, Ptr<Options> options)
      : Layer(graph, options) {}

  void tie(const std::string& param, const std::string& tied) {
    tiedParams_[param] = graph_->get(tied);
  }

  void tie_transposed(const std::string& param, const std::string& tied) {
    tiedParams_[param] = transpose(graph_->get(tied));
  }

  Expr apply(const std::vector<Expr>& inputs) {
    ABORT_IF(inputs.empty(), "No inputs");

    if(inputs.size() == 1)
      return apply(inputs[0]);

    auto name = opt<std::string>("prefix");
    auto dim = opt<int>("dim");

    auto layerNorm = opt<bool>("layer-normalization", false);
    auto nematusNorm = opt<bool>("nematus-normalization", false);
    auto activation = opt<act>("activation", act::linear);

    auto g = graph_;

    params_ = {};
    std::vector<Expr> outputs;
    size_t i = 0;
    for(auto&& in : inputs) {
      Expr W;
      std::string nameW = "W" + std::to_string(i);
      if(tiedParams_.count(nameW))
        W = tiedParams_[nameW];
      else
        W = g->param(name + "_" + nameW,
                     {in->shape()[-1], dim},
                     keywords::init = inits::glorot_uniform);

      Expr b;
      std::string nameB = "b" + std::to_string(i);
      if(tiedParams_.count(nameB))
        b = tiedParams_[nameB];
      else
        b = g->param(
            name + "_" + nameB, {1, dim}, keywords::init = inits::zeros);

      params_.push_back(W);
      params_.push_back(b);

      if(layerNorm) {
        if(nematusNorm) {
          auto ln_s = g->param(name + "_ln_s" + std::to_string(i),
                               {1, dim},
                               keywords::init = inits::from_value(1.f));
          auto ln_b = g->param(name + "_ln_b" + std::to_string(i),
                               {1, dim},
                               keywords::init = inits::zeros);

          outputs.push_back(
              layer_norm(affine(in, W, b), ln_s, ln_b, NEMATUS_LN_EPS));
        } else {
          auto gamma = g->param(name + "_gamma" + std::to_string(i),
                                {1, dim},
                                keywords::init = inits::from_value(1.0));

          params_.push_back(gamma);
          outputs.push_back(layer_norm(dot(in, W), gamma, b));
        }

      } else {
        outputs.push_back(affine(in, W, b));
      }
      i++;
    }

    switch(activation) {
      case act::linear: return plus(outputs);
      case act::tanh: return tanh(outputs);
      case act::logit: return logit(outputs);
      case act::ReLU: return relu(outputs);
      case act::LeakyReLU: return leakyrelu(outputs);
      case act::PReLU: return prelu(outputs);
      case act::swish: return swish(outputs);
      default: return plus(outputs);
    }
  };

  Expr apply(Expr input) {
    auto g = graph_;

    auto name = options_->get<std::string>("prefix");
    auto dim = options_->get<int>("dim");

    auto layerNorm = options_->get<bool>("layer-normalization", false);
    auto nematusNorm = opt<bool>("nematus-normalization", false);
    auto activation = options_->get<act>("activation", act::linear);

    Expr W;
    std::string nameW = "W";
    if(tiedParams_.count(nameW))
      W = tiedParams_[nameW];
    else
      W = g->param(name + "_" + nameW,
                   {input->shape()[-1], dim},
                   keywords::init = inits::glorot_uniform);

    Expr b;
    std::string nameB = "b";
    if(tiedParams_.count(nameB))
      b = tiedParams_[nameB];
    else
      b = g->param(name + "_" + nameB, {1, dim}, keywords::init = inits::zeros);

    params_ = {W, b};

    Expr out;
    if(layerNorm) {
      if(nematusNorm) {
        auto ln_s = g->param(
            name + "_ln_s", {1, dim}, keywords::init = inits::from_value(1.f));
        auto ln_b
            = g->param(name + "_ln_b", {1, dim}, keywords::init = inits::zeros);

        out = layer_norm(affine(input, W, b), ln_s, ln_b, NEMATUS_LN_EPS);
      } else {
        auto gamma = g->param(
            name + "_gamma", {1, dim}, keywords::init = inits::from_value(1.0));

        params_.push_back(gamma);
        out = layer_norm(dot(input, W), gamma, b);
      }
    } else {
      out = affine(input, W, b);
    }

    switch(activation) {
      case act::linear: return out;
      case act::tanh: return tanh(out);
      case act::logit: return logit(out);
      case act::ReLU: return relu(out);
      case act::LeakyReLU: return leakyrelu(out);
      case act::PReLU: return prelu(out);
      case act::swish: return swish(out);
      default: return out;
    }
  }
};

}  // namespace mlp

struct EmbeddingFactory : public Factory {
  EmbeddingFactory(Ptr<ExpressionGraph> graph) : Factory(graph) {}

  Expr construct() {
    std::string name = opt<std::string>("prefix");
    int dimVoc = opt<int>("dimVocab");
    int dimEmb = opt<int>("dimEmb");

    bool fixed = opt<bool>("fixed", false);

    //std::function<void(Tensor)> initFunc = inits::glorot_uniform;
    auto initFunc = inits::glorot_uniform;
    if(options_->has("embFile")) {
      std::string file = opt<std::string>("embFile");
      if(!file.empty()) {
        bool norm = opt<bool>("normalization", false);
        initFunc = inits::from_word2vec(file, dimVoc, dimEmb, norm);
      }
    }

    return graph_->param(name,
                         {dimVoc, dimEmb},
                         keywords::init = initFunc,
                         keywords::fixed = fixed);
  }
};

typedef Accumulator<EmbeddingFactory> embedding;

Expr Cost(Expr logits,
          Expr indices,
          Expr mask,
          std::string costType = "cross-entropy",
          float smoothing = 0);
}
