#ifndef MULTIVERSO_TABLE_FACTORY_H_
#define MULTIVERSO_TABLE_FACTORY_H_

#include "multiverso/table_interface.h"
#include "multiverso/zoo.h"

#include <string>

namespace multiverso {

namespace table_factory {

void FreeServerTables();
void PushServerTable(ServerTable*table);

template <typename OptionType>
typename OptionType::WorkerTableType* CreateTable(const OptionType& option) {
  if (Zoo::Get()->server_rank() >= 0) {
    PushServerTable(
      new typename OptionType::ServerTableType(option));
  }
  if (Zoo::Get()->worker_rank() >= 0) {
    return new typename OptionType::WorkerTableType(option);
  }
  return nullptr;
}

} // namespace table_factory

} // namespace multiverso

#endif // MULTIVERSO_TABLE_FACTORY_H_
