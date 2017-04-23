#ifndef LOGREG_UTIL_HELPER_H_
#define LOGREG_UTIL_HELPER_H_

#include "data_type.h"

namespace logreg {

template <typename EleType>
EleType** CreateMatrix(int num_row, int num_col) {
  EleType **matrix = new EleType*[num_row];
  for (int i = 0; i < num_row; ++i)
    matrix[i] = new EleType[num_col];
  return matrix;
}

template <typename EleType>
void FreeMatrix(int num_row, EleType**matrix) {
  for (int i = 0; i < num_row; ++i)
    delete[]matrix[i];
  delete[]matrix;
}

template <typename EleType>
EleType Dot(size_t offset, DataBlock<EleType>*matrix, Sample<EleType>*sample) {
  EleType sum = 0;
  int size = static_cast<int>(sample->values.size());
  if (matrix->sparse()) {
    DEBUG_CHECK(sample->keys.size() == sample->values.size());
    for (int i = 0; i < size; ++i) {
      EleType* pval = matrix->Get(sample->keys[i] + offset);
      sum += (pval == nullptr ? 0 : (sample->values[i] * (*pval)));
    }
  } else {
    EleType*rawa = static_cast<EleType*>(matrix->raw()) + offset;
    EleType*rawb = sample->values.data();
    for (int i = 0; i < size; ++i) {
      sum += rawa[i] * rawb[i];
    }
  }
  return sum;
}

template <typename EleType>
inline EleType* MatrixRow(EleType*matrix, int row_id, size_t num_col) {
  return matrix + row_id * num_col;
}

template <typename EleType>
Sample<EleType>** CeateSamples(int num, size_t size, bool sparse) {
  Sample<EleType>**samples = new Sample<EleType>*[num];
  for (int i = 0; i < num; ++i) {
    samples[i] = new Sample<EleType>(sparse, size);
  }
  return samples;
}

template <typename EleType>
void FreeSamples(int num, Sample<EleType>**samples) {
  for (int i = 0; i < num; ++i) {
    delete samples[i];
  }
  delete[]samples;
}

#define DECLARE_TEMPLATE_CLASS_WITH_BASIC_TYPE(name)  \
  template class name<int>;                           \
  template class name<float>;                         \
  template class name<double>;

}  // namespace logreg

#endif  // LOGREG_UTIL_HELPER_H_
