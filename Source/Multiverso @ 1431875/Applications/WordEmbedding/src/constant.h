#ifndef WORDEMBEDDING_CONSTANT_H_
#define WORDEMBEDDING_CONSTANT_H_

/*!
* \file constant.h
* \brief The index of parameter tables and some constant.
*/
#include <cstdint>

namespace wordembedding {
  typedef int64_t int64;
  typedef uint64_t uint64;
  typedef float real;

  //multiverso table id
  const int kInputEmbeddingTableId = 0;
  const int kEmbeddingOutputTableId = 1;
  const int kSumGradient2IETableId = 2;
  const int kSumGradient2EOTableId = 3;
  const int kWordCountId = 4;

  const int kTableSize = (int)1e8;

  const int kMaxWordSize = 901;
  const int kMaxCodeLength = 100;
  const int kMaxString = 500;
  const int kMaxSentenceLength = 1000;
  const int kMaxExp = 6;

  const int kExpTableSize = 1000;
  const int kSaveBatch = 100000;
}
#endif
