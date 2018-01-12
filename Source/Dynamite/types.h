#pragma once

#include <iostream>
#include <vector>

#include "common/definitions.h"
#include "common/options.h"
#include "graph/expression_graph.h"

namespace marian {
namespace rnn {

struct State {
  Expr output;
  Expr cell;

  State select(const std::vector<size_t>& indices, int beamSize) {
    output = atleast_4d(output);
    if(cell)
      cell = atleast_4d(cell);

    int dimDepth = output->shape()[-1];
    int dimTime  = output->shape()[-3];

    int dimBatch = indices.size() / beamSize;

    if(cell) {
      return State{
          reshape(rows(flatten_2d(output), indices),
                  {beamSize, dimTime, dimBatch, dimDepth}),
          reshape(rows(flatten_2d(cell), indices),
                  {beamSize, dimTime, dimBatch, dimDepth})};
    } else {
      return State{
        reshape(rows(flatten_2d(output), indices),
                {beamSize, dimTime, dimBatch, dimDepth}),
        nullptr};
    }
  }
};

class States {
private:
  std::vector<State> states_;

public:
  States() {}
  States(const std::vector<State>& states) : states_(states) {}
  States(size_t num, State state) : states_(num, state) {}

  auto begin() -> decltype(states_.begin()) { return states_.begin(); }
  auto end() -> decltype(states_.begin()) { return states_.end(); }

  Expr outputs() {
    std::vector<Expr> outputs;
    for(auto s : states_)
      outputs.push_back(atleast_3d(s.output));
    if(outputs.size() > 1)
      return concatenate(outputs, keywords::axis = -3);
    else
      return outputs[0];
  }

  State& operator[](size_t i) { return states_[i]; };
  const State& operator[](size_t i) const { return states_[i]; };

  State& back() { return states_.back(); }
  const State& back() const { return states_.back(); }

  State& front() { return states_.front(); }
  const State& front() const { return states_.front(); }

  size_t size() const { return states_.size(); };

  void push_back(const State& state) { states_.push_back(state); }

  States select(const std::vector<size_t>& indices, int beamSize) {
    States selected;
    for(auto& state : states_)
      selected.push_back(state.select(indices, beamSize));
    return selected;
  }

  void reverse() { std::reverse(states_.begin(), states_.end()); }

  void clear() { states_.clear(); }
};

class Cell;
struct CellInput;

class Stackable : public std::enable_shared_from_this<Stackable> {
protected:
  Ptr<Options> options_;

public:
  Stackable(Ptr<Options> options) : options_(options) {}

  // required for dynamic_pointer_cast to detect polymorphism
  virtual ~Stackable() {}

  template <typename Cast>
  inline Ptr<Cast> as() {
    return std::dynamic_pointer_cast<Cast>(shared_from_this());
  }

  template <typename Cast>
  inline bool is() {
    return as<Cast>() != nullptr;
  }

  Ptr<Options> getOptions() { return options_; }

  template <typename T>
  T opt(const std::string& key) {
    return options_->get<T>(key);
  }

  template <typename T>
  T opt(const std::string& key, T defaultValue) {
    return options_->get<T>(key, defaultValue);
  }

  virtual void clear() = 0;
};

class CellInput : public Stackable {
public:
  CellInput(Ptr<Options> options) : Stackable(options) {}

  virtual Expr apply(State) = 0;
  virtual int dimOutput() = 0;
};

class RNN;

class Cell : public Stackable {
protected:
  std::vector<std::function<Expr(Ptr<rnn::RNN>)>> lazyInputs_;

public:
  Cell(Ptr<Options> options) : Stackable(options) {}

  State apply(std::vector<Expr> inputs, State state, Expr mask = nullptr) {
    return applyState(applyInput(inputs), state, mask);
  }

  virtual std::vector<Expr> getLazyInputs(Ptr<rnn::RNN> parent) {
    std::vector<Expr> inputs;
    for(auto lazy : lazyInputs_)
      inputs.push_back(lazy(parent));
    return inputs;
  }

  virtual void setLazyInputs(
      std::vector<std::function<Expr(Ptr<rnn::RNN>)>> lazy) {
    lazyInputs_ = lazy;
  }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) = 0;
  virtual State applyState(std::vector<Expr>, State, Expr = nullptr) = 0;

  virtual void clear() {}
};

class MultiCellInput : public CellInput {
protected:
  std::vector<Ptr<CellInput>> inputs_;

public:
  MultiCellInput(const std::vector<Ptr<CellInput>>& inputs,
                 Ptr<Options> options)
      : CellInput(options), inputs_(inputs) {}

  void push_back(Ptr<CellInput> input) { inputs_.push_back(input); }

  virtual Expr apply(State state) {
    std::vector<Expr> outputs;
    for(auto input : inputs_)
      outputs.push_back(input->apply(state));

    if(outputs.size() > 1)
      return concatenate(outputs, keywords::axis = -1);
    else
      return outputs[0];
  }

  virtual int dimOutput() {
    int sum = 0;
    for(auto input : inputs_)
      sum += input->dimOutput();
    return sum;
  }

  virtual void clear() {
    for(auto i : inputs_)
      i->clear();
  }
};

class StackedCell : public Cell {
protected:
  std::vector<Ptr<Stackable>> stackables_;
  std::vector<Expr> lastInputs_;

public:
  StackedCell(Ptr<ExpressionGraph>, Ptr<Options> options) : Cell(options) {}

  StackedCell(Ptr<ExpressionGraph>,
              Ptr<Options> options,
              const std::vector<Ptr<Stackable>>& stackables)
      : Cell(options), stackables_(stackables) {}

  void push_back(Ptr<Stackable> stackable) { stackables_.push_back(stackable); }

  virtual std::vector<Expr> applyInput(std::vector<Expr> inputs) {
    // lastInputs_ = inputs;
    return stackables_[0]->as<Cell>()->applyInput(inputs);
  }

  virtual State applyState(std::vector<Expr> mappedInputs,
                           State state,
                           Expr mask = nullptr) {
    State hidden
        = stackables_[0]->as<Cell>()->applyState(mappedInputs, state, mask);
    ;

    for(int i = 1; i < stackables_.size(); ++i) {
      if(stackables_[i]->is<Cell>()) {
        auto hiddenNext
            = stackables_[i]->as<Cell>()->apply(lastInputs_, hidden, mask);
        lastInputs_.clear();
        hidden = hiddenNext;
      } else {
        lastInputs_.push_back(stackables_[i]->as<CellInput>()->apply(hidden));
        // lastInputs_ = { stackables_[i]->as<CellInput>()->apply(hidden) };
      }
    }

    return hidden;
  };

  Ptr<Stackable> operator[](int i) { return stackables_[i]; }

  Ptr<Stackable> at(int i) { return stackables_[i]; }

  virtual void clear() {
    for(auto s : stackables_)
      s->clear();
  }

  virtual std::vector<Expr> getLazyInputs(Ptr<rnn::RNN> parent) {
    ABORT_IF(!stackables_[0]->is<Cell>(),
             "First stackable should be of type Cell");
    return stackables_[0]->as<Cell>()->getLazyInputs(parent);
  }

  virtual void setLazyInputs(
      std::vector<std::function<Expr(Ptr<rnn::RNN>)>> lazy) {
    ABORT_IF(!stackables_[0]->is<Cell>(),
             "First stackable should be of type Cell");
    stackables_[0]->as<Cell>()->setLazyInputs(lazy);
  }
};
}
}
