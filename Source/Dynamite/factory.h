#pragma once

//#include "common/options.h"
//#include "graph/expression_graph.h"

namespace marian {

class Factory : public std::enable_shared_from_this<Factory> {
protected:
  Ptr<Options> options_;
  Ptr<ExpressionGraph> graph_;

public:
  Factory(Ptr<ExpressionGraph> graph)
      : options_(New<Options>()), graph_(graph) {}

  virtual ~Factory() {}

  Ptr<Options> getOptions() { return options_; }

  std::string str() { return options_->str(); }

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, T defaultValue) {
    return options_->get<T>(key, defaultValue);
  }
};

template <class BaseFactory>
class Accumulator : public BaseFactory {
  typedef BaseFactory Factory;
public:
  Accumulator() : Factory(nullptr) {}
  Accumulator(Ptr<ExpressionGraph> graph) : Factory(graph) {}
  Accumulator(const Factory& factory) : Factory(factory) {}
  Accumulator(const Accumulator&) = default;
  Accumulator(Accumulator&&) = default;

  template <typename T>
  Accumulator& operator()(const std::string& key, T value) {
    Factory::getOptions()->set(key, value);
    return *this;
  }

  //Accumulator& operator()(const std::string& yaml) {
  //  Factory::getOptions()->parse(yaml);
  //  return *this;
  //}

  //Accumulator& operator()(YAML::Node yaml) {
  //  Factory::getOptions()->merge(yaml);
  //  return *this;
  //}

  Accumulator& operator()(Ptr<Options> options) {
    Factory::getOptions()->merge(options);
    return *this;
  }

  //Accumulator& operator()(Ptr<Config> config) {
  //  Factory::getOptions()->merge(config->get());
  //  return *this;
  //}

  Accumulator<Factory> clone() {
    return Accumulator<Factory>(Factory::clone());
  }
};
}
