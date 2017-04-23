#ifndef LOGREG_DATA_BLOCK_H_
#define LOGREG_DATA_BLOCK_H_

#include <math.h>
#include <vector>

#include "util/hopscotch_hash.h"
#include "util/log.h"

namespace logreg {

// structure for ftrl
template <typename EleType>
struct FTRLGradient {
  inline FTRLGradient(EleType z, EleType n) :
  delta_z(z), delta_n(n) {
  }
  inline FTRLGradient() : delta_z(0), delta_n(0) {}
  inline bool operator!=(const int& zero) {
    DEBUG_CHECK(zero == 0);
    return delta_z != 0 || delta_n != 0;
  }
  inline void operator-=(const FTRLGradient&d) {
    delta_z -= d.delta_z;
    delta_n -= d.delta_n;
  }
  EleType delta_z;
  EleType delta_n;
};  // struct FTRLGradient

template <typename EleType>
struct FTRLEntry {
  inline FTRLEntry() : z(0), n(0), sqrtn(0) {}
  inline FTRLEntry(EleType z, EleType n) :
  z(z), n(n),
  sqrtn((EleType)sqrt(n)) {
    DEBUG_CHECK(n >= 0);
  }
  explicit inline FTRLEntry(FTRLGradient<EleType>* g) :
  z(g->delta_z), n(g->delta_n),
  sqrtn((EleType)sqrt(g->delta_n)) {
    DEBUG_CHECK(n >= 0);
  }
  inline bool operator!=(const FTRLEntry&en) {
    return z != en.z || n != en.n;
  }
  inline bool operator!=(const int& zero) {
    DEBUG_CHECK(zero == 0);
    return z != 0 || n != 0;
  }
  EleType z;
  EleType n;
  EleType sqrtn;
};  // struct FTRLEntry

template<typename EleType>
struct Sample {
  Sample(bool sparse, size_t data_size)  {
    DEBUG_CHECK(sparse || data_size != 0);

    if (!sparse) {
      values.reserve(data_size);
    }
  }

  int label;
  std::vector<size_t> keys;
  std::vector<EleType> values;
};  // struct Sample


template <typename EleType>
class SparseBlock;
template <typename EleType>
class DenseBlock;

// base type for data block
template <typename EleType>
class DataBlock {
public:
  explicit DataBlock(bool sparse) : sparse_(sparse) {}
  virtual ~DataBlock() = default;
  // set value
  virtual void Set(size_t key, EleType value) = 0;
  virtual void Set(size_t key, EleType* value) = 0;
  virtual void Set(DataBlock* block) = 0;
  // return nullptr if not found
  virtual EleType* Get(size_t key) = 0;
  virtual size_t size() const = 0;
  virtual size_t capacity() const = 0;
  // set all value to zero if dense
  virtual void Clear() = 0;
  virtual void *raw() = 0;

  bool sparse() const { return sparse_; }

  static DataBlock<EleType>* GetBlock(bool sparse, size_t size = 0) {
    if (sparse) {
      return new SparseBlock<EleType>();
    } else {
      DEBUG_CHECK(size != 0);
      return new DenseBlock<EleType>(size);
    }
  }

protected:
  bool sparse_;
};  // class DataBlock


template <typename EleType>
class SparseBlockIter;
template <typename EleType>
class DenseBlockIter;

template <typename EleType>
class DataBlockIter {
public:
  explicit DataBlockIter(DataBlock<EleType>* block) : data_block_(block) {}
  virtual ~DataBlockIter() = default;
  // should call Next first
  virtual bool Next() = 0;
  virtual EleType* Value() = 0;
  virtual size_t Key() = 0;
  virtual void Reset() = 0;
  // factory method for new iterator
  static DataBlockIter<EleType>* Get(DataBlock<EleType>* block) {
    if (block->sparse()) {
      return new SparseBlockIter<EleType>(block);
    } else {
      return new DenseBlockIter<EleType>(block);
    }
  }
protected:
  DataBlock<EleType>* data_block_;
};  // class DataBlockIter

template <typename EleType>
class DenseBlockIter : public DataBlockIter<EleType> {
public:
  explicit DenseBlockIter(DataBlock<EleType>* block)
    : DataBlockIter<EleType>(block), idx_(-1) {
    DEBUG_CHECK(block->sparse() == false);
  }
  ~DenseBlockIter() = default;
  inline bool Next() {
    return (++idx_ == this->data_block_->size());
  }
  inline EleType* Value() {
    return this->data_block_->Get(idx_);
  }
  inline size_t Key() {
    return idx_;
  }
  inline void Reset() {
    idx_ = -1;
  }
protected:
  size_t idx_;
};  // class DenseBlockIter

template <typename EleType>
class DenseBlock : public DataBlock<EleType> {
public:
  explicit DenseBlock(size_t size) : DataBlock<EleType>(false), size_(size) {
    values_ = new EleType[size];
  }
  ~DenseBlock() {
    delete[]values_;
  }
  inline void Set(size_t key, EleType *value) {
    DEBUG_CHECK(key < size_);
    values_[key] = *value;
  }
  inline void Set(size_t key, EleType value) {
    DEBUG_CHECK(key < size_);
    values_[key] = value;
  }
  inline void Set(DataBlock<EleType>* block) {
    DEBUG_CHECK(block->sparse() == false);
    DEBUG_CHECK(block->size() == size_);
    for (size_t i = 0; i < size_; ++i) {
      values_[i] = *block->Get(i);
    }
  }
  inline void Clear() {
    memset(values_, 0, sizeof(EleType) * size_);
  }
  inline EleType* Get(size_t key) {
    return values_ + key; 
  }
  inline size_t size() const { return size_; }
  inline size_t capacity() const { return size_; }
  inline void *raw() {
    return static_cast<void*>(values_);
  }

private:
  size_t size_;
  EleType* values_;
};  // class DenseBlock

template <typename EleType>
class SparseBlockIter : public DataBlockIter<EleType> {
public:
  explicit SparseBlockIter(DataBlock<EleType>* block)
    : DataBlockIter<EleType>(block) {
    DEBUG_CHECK(block->sparse() == true);
    iter_ = static_cast<Hopscotch<EleType>*>(
      static_cast<SparseBlock<EleType>*>(block)->raw())
      ->GetIter();
  }
  ~SparseBlockIter() {
    delete iter_;
  }
  inline bool Next() {
    return iter_->Next();
  }
  inline EleType* Value() {
    return iter_->value();
  }
  inline size_t Key() {
    return iter_->key();
  }
  inline void Reset() {
    iter_->Reset();
  }

protected:
  typename Hopscotch<EleType>::Iter *iter_;
};  // SparseBlockIter

template <typename EleType>
class SparseBlock : public DataBlock<EleType> {
public:
  explicit SparseBlock(size_t capacity = 0) : DataBlock<EleType>(true) {
    values_ = new Hopscotch<EleType>(capacity);
  }
  ~SparseBlock() {
    delete values_;
  }
  inline void Set(DataBlock<EleType>* block) {
    DEBUG_CHECK(block->sparse());
    SparseBlockIter<EleType> iter(block);
    while (iter.Next()) {
      values_->Set(iter.Key(), iter.Value());
    }
  }
  inline void Set(size_t key, EleType *value) { 
    values_->Set(key, value); 
  }
  inline void Set(size_t key, EleType value) {
    values_->Set(key, &value);
  }
  inline EleType* Get(size_t key) { 
    return values_->Get(key);
  }
  inline size_t size() const {
    return values_->size(); 
  }
  inline size_t capacity() const {
    return values_->capacity();
  }
  inline void Clear() { 
    values_->Clear(); 
  }
  inline void *raw() {
    return static_cast<void*>(values_);
  }

private:
  Hopscotch<EleType> *values_;
};  // class SparseBlock

}  // namespace logreg

#endif  // LOGREG_DATA_BLOCK_H_
