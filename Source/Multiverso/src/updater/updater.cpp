#include "multiverso/updater/updater.h"
// TODO(qiwye) to make this a option in CMakelist
//#define ENABLE_DCASGD

#include "multiverso/updater/adagrad_updater.h"
#include "multiverso/updater/momentum_updater.h"
#ifdef ENABLE_DCASGD
#include "multiverso/updater/dcasgd/dcasgd_updater.h"
#endif
#include "multiverso/updater/sgd_updater.h"
#include "multiverso/util/configure.h"
#include "multiverso/util/log.h"


namespace multiverso {

MV_DEFINE_string(updater_type, "default", "multiverso server updater type");
MV_DEFINE_int(omp_threads, 4 , "#theads used by openMP for updater");
#ifdef ENABLE_DCASGD
MV_DEFINE_bool(is_pipelined, false, "Only used for CNTK - DCASGD");
#endif

template <typename T>
void Updater<T>::Update(size_t num_element, T* data, T* delta,
                        AddOption*, size_t offset) {
  // parallelism with openMP
  #pragma omp parallel for schedule(static) num_threads(MV_CONFIG_omp_threads)
  for (int i = 0; i < num_element; ++i) {
    data[i + offset] += delta[i];
  }
}

template <typename T>
void Updater<T>::Access(size_t num_element, T* data, T* blob_data,
  size_t offset , AddOption*) {
  // copy data from data to blob
  memcpy(blob_data, data + offset, num_element * sizeof(T));
}

// Gradient-based updater in only for numerical table
// For simple int table, just using simple updater
template<>
Updater<int>* Updater<int>::GetUpdater(size_t) {
  return new Updater<int>();
}

template <typename T>
Updater<T>* Updater<T>::GetUpdater(size_t size) {
  std::string type = MV_CONFIG_updater_type;
  if (type == "sgd") return new SGDUpdater<T>(size);
  if (type == "adagrad") return new AdaGradUpdater<T>(size);
  if (type == "momentum_sgd") return new MomentumUpdater<T>(size);
#ifdef ENABLE_DCASGD
  if (type == "dcasgd") return new DCASGDUpdater<T>(size, MV_CONFIG_is_pipelined);
#endif
  // Default: simple updater
  return new Updater<T>();
}

MV_INSTANTIATE_CLASS_WITH_BASE_TYPE(Updater);

}