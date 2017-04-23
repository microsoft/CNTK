#include "multiverso/table_factory.h"

#include "multiverso/table/array_table.h"
#include "multiverso/table/matrix_table.h"

namespace multiverso {

namespace table_factory {
std::vector<ServerTable*> server_tables;

void FreeServerTables() {
  for (auto table : server_tables) {
    delete table;
  }
  server_tables.clear();
}

void PushServerTable(ServerTable*table) {
  server_tables.push_back(table);
}

} // namespace table_factory

} // namespace multiverso