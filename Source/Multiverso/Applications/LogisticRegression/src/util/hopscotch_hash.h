#ifndef LOGREG_UTIL_HOPSCOTCH_HASH_H_
#define LOGREG_UTIL_HOPSCOTCH_HASH_H_

#include <string.h>

#include "util/log.h"
#include "multiverso/util/allocator.h"

namespace logreg {

template <typename EleType>
class Hopscotch {
private:
  using uint8 = unsigned char;
  struct Bucket {
    // first element offset in this bucket
    // offset start from 1, 0 for none
    uint8 first; 
    // next element offset for element in this position
    uint8 next;
    // if this bucket is used to store element
    bool use;
    // elements
    size_t key;
    EleType val;
  };  // struct Bucket

public:
  explicit Hopscotch(size_t size = min_capacity_);
  ~Hopscotch();
  // return value if exist, else nullptr
  EleType* Get(size_t key);
  // insert if not exist, else set value
  void Set(size_t key, EleType *val);
  // remove if exist
  void Remove(size_t key);
  
  void Clear();
  
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

public:
  struct Iter {
    Iter(Bucket* buckets, size_t capacity) :
    initial(buckets), bucket(buckets) {
      end = bucket + capacity;
      if (!bucket->use) {
        Next();
      }
      --bucket;
    }
    inline EleType* value() { return &bucket->val; }
    inline size_t key() { return bucket->key; }
    // false when hit end
    inline bool Next() {
      while (++bucket != end) {
        if (bucket->use) {
          return true;
        }
      }
      return false;
    }
    inline void Reset() {
      bucket = initial;
      if (!bucket->use) {
        Next();
      }
      --bucket;
    }

  private:
    Bucket* initial;
    Bucket* bucket;
    Bucket* end;
  };  // struct Iter
  Iter* GetIter() {
    return new Iter(buckets_, capacity_);
  }

private:
  // hash function
  size_t HashVal(size_t key);

  Bucket* Home(size_t key);
  Bucket* First(Bucket* home);
  Bucket* Next(Bucket* bucket);
  Bucket* Find(Bucket* home, size_t key);
  Bucket* Prev(Bucket* home, Bucket* bucket);
  Bucket* Displace(Bucket* home, Bucket* empty);
  void Insert(Bucket* home, size_t key, EleType *val);
  // resize for capacity * 2 and rehash
  void Grow();

private:
  static const uint8 neighbour_size_ = 32;
  static const size_t min_capacity_ = 256;
  static const size_t max_capacity_ = (size_t)-neighbour_size_;

  Bucket* buckets_;
  size_t size_;
  size_t capacity_;
  size_t pow2_capacity_;
};

#define BUCKET typename Hopscotch<EleType>::Bucket

template <typename EleType>
Hopscotch<EleType>::Hopscotch(size_t size) :
size_(0) {
  // make size pow of 2
  if (size < min_capacity_) {
    size = min_capacity_; 
  } else {
    size -= 1;
    size |= size >> 32;
    size |= size >> 16;
    size |= size >> 8;
    size |= size >> 4;
    size |= size >> 2;
    size |= size >> 1;
    size += 1;
  }
  LR_CHECK(size <= max_capacity_);

  pow2_capacity_ = size;
  capacity_ = size + neighbour_size_;
  
  buckets_ = reinterpret_cast<Bucket*>(multiverso::Allocator::Get()
    ->Alloc(capacity_ * sizeof(Bucket)));
  memset(buckets_, 0, sizeof(Bucket)* capacity_);
}

template <typename EleType>
Hopscotch<EleType>::~Hopscotch() {
  // Log::Write(Debug, "~Hopscotch hash, final size %lld, capacity %lld\n",
    // size_, capacity_);
  multiverso::Allocator::Get()->Free(reinterpret_cast<char*>(buckets_));
}

template <typename EleType>
inline BUCKET* Hopscotch<EleType>::Home(size_t key) {
  return HashVal(key) + buckets_;
}

template <typename EleType>
inline BUCKET* Hopscotch<EleType>::First(Bucket* home) {
#ifdef LOGLEVEL_DEBUG
  if (home->first == 0) {
    return nullptr;
  } else {
    LR_CHECK((home + home->first - 1)->use);
    return (home + home->first - 1);
  }
#else
  return home->first == 0 ? nullptr : (home + home->first - 1);
#endif
}

template <typename EleType>
inline BUCKET* Hopscotch<EleType>::Next(Bucket* bucket) {
#ifdef LOGLEVEL_DEBUG
  if (bucket->next == 0) {
    return nullptr;
  } else {
    LR_CHECK((bucket + bucket->next - 1)->use);
    return (bucket + bucket->next - 1);
  }
#else
  return bucket->next == 0 ? nullptr : (bucket + bucket->next - 1);
#endif
}

template <typename EleType>
inline BUCKET* Hopscotch<EleType>::Prev(Bucket* home, Bucket* bucket) {
  Bucket* move = First(home);
  Bucket* prev = nullptr;
  while (move != nullptr && move < bucket) {
    prev = move;
    move = Next(move);
  }
  return prev;
}

template <typename EleType>
void Hopscotch<EleType>::Clear() {
  if (size_ == 0) {
    return;
  }
  // reset value
  memset(buckets_, 0, sizeof(Bucket)* capacity_);
  size_ = 0;
}

template <typename EleType>
void Hopscotch<EleType>::Grow() {
  size_t old_capacity = capacity_;
  Bucket *old_buckets = buckets_;
  
  pow2_capacity_ = pow2_capacity_ << 1;
  capacity_ = pow2_capacity_ + neighbour_size_;
  // reallocate
  buckets_ = reinterpret_cast<Bucket*>(multiverso::Allocator::Get()
    ->Alloc(capacity_ * sizeof(Bucket)));
  memset(buckets_, 0, sizeof(Bucket)* capacity_);

  // rehash
  Bucket *move = old_buckets;
  Bucket *move_end = old_buckets + old_capacity;
  while (move < move_end) {
    if (move->use) {
      Insert(Home(move->key), move->key, &move->val);
    }
    ++move;
  }

  multiverso::Allocator::Get()->Free(reinterpret_cast<char*>(old_buckets));
}

template <typename EleType>
void Hopscotch<EleType>::Set(size_t key, EleType *val) {
  Bucket* home = Home(key);
  Bucket* bucket = Find(home, key);
  if (bucket != nullptr) {
    bucket->val = *val;
    return;
  }
  // insert for not found
  ++size_;
  Insert(home, key, val);
}

template <typename EleType>
void Hopscotch<EleType>::Remove(size_t key) {
  Bucket* home = Home(key);
  Bucket* bucket = Find(home, key);
  if (bucket != nullptr) {
    // update list info
    Bucket* prev = Prev(home, bucket);
    Bucket* next = Next(bucket);
    if (prev == nullptr) {
      home->first = (uint8)(next == nullptr ? 0 : (next - home + 1));
    } else {
      prev->next = (uint8)(next == nullptr ? 0 : (next - prev + 1));
    }

    bucket->use = false;
    --size_;

    // do compact
    while (bucket->first != 0) {
      Bucket* move = First(bucket);
      prev = bucket;
      while (move->next != 0) {
        prev = move;
        move = Next(move);
      }
      // move last element info bucket position
      bucket->key = move->key;
      bucket->val = move->val;
      bucket->use = true;
      bucket->next = bucket->first;
      bucket->first = 1;

      prev->next = 0;
      DEBUG_CHECK(bucket->next != (uint8)1);
      move->use = false;
      bucket = move;
    }
  }
}

template <typename EleType>
void Hopscotch<EleType>::Insert(Bucket* home, size_t key, EleType *val) {
  Bucket* move = home;
  Bucket* end = home + (neighbour_size_ << 3);
  end = end < buckets_ + capacity_ ? end : buckets_ + capacity_;
  while (true) {
    // too far or end of buckets
    if (move == end) {
      Grow();
      Insert(Home(key), key, val);
      return;
    }
    // find empty
    if (!move->use) {
      break;
    }
    ++move;
  }

  // outside neighbor
  if ((move - home) >= neighbour_size_) {
    // try displace
    move = Displace(home, move);
    // resize when fail
    if (move == nullptr) {
      Grow();
      return Insert(Home(key), key, val);
    }
  }
  DEBUG_CHECK(move->use == false);
  // place element in move
  move->key = key;
  move->val = *val;
  move->use = true;
  // update list info
  Bucket* prev = Prev(home, move);
  if (prev == nullptr) {
    home->first = (uint8)(move - home + 1);
    move->next = 0;
  } else {
    Bucket* next = Next(prev);
    prev->next = (uint8)(move - prev + 1);
    move->next = next == nullptr ? 0 : (uint8)(next - move + 1);
    DEBUG_CHECK(move->next != (uint8)1);
  }
}

template <typename EleType>
BUCKET* Hopscotch<EleType>::Displace(Bucket* home, Bucket* empty) {
  Bucket* start = empty - neighbour_size_;
  Bucket* within = home + neighbour_size_;
  Bucket* candidate = empty - 1;
  Bucket* candi_home;

  while (candidate > home) {
    candi_home = Home(candidate->key);

    if (candi_home > start) {
      // move the element in candidate to empty place
      empty->use = true;
      empty->key = candidate->key;
      empty->val = candidate->val;
      // update list
      uint8 diff = (uint8)(empty - candidate);
      Bucket* prev = Prev(candi_home, candidate);
      (prev == nullptr ? candi_home->first : prev->next) += diff;
      empty->next = candidate->next - (candidate->next == 0 ? 0 : diff);
      DEBUG_CHECK(empty->next != (uint8)1);
      // now candidate is empty
      candidate->use = false;
      empty = candidate;
      // update start
      start = empty - neighbour_size_;
      // valid empty position
      if (empty < within) {
        return empty;
      }
    }
    --candidate;
  }

  DEBUG_CHECK(!empty->use);
  // not found
  return nullptr;
}

template <typename EleType>
EleType* Hopscotch<EleType>::Get(size_t key) {
  Bucket* ret = Find(Home(key), key);
  return (ret == nullptr ? nullptr : (&ret->val));
}

template <typename EleType>
inline BUCKET* Hopscotch<EleType>::Find(Bucket* home, size_t key) {
  Bucket *move = First(home);
  while (move != nullptr) {
    if (move->key == key) {
      return move;
    }
    move = Next(move);
  }
  return nullptr;
}

// simple hash
template <typename EleType>
inline size_t Hopscotch<EleType>::HashVal(size_t key) {
  return key & (pow2_capacity_ - 1);
}

}  // namespace logreg

#endif  // LOGREG_UTIL_HOPSCOTCH_HASH_H_
