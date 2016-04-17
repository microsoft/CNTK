#ifndef MULTIVERSO_SERVER_H_
#define MULTIVERSO_SERVER_H_

#include <string>
#include <vector>

#include "multiverso/actor.h"

namespace multiverso {

class ServerTable;

class Server : public Actor {
public:
  Server();
  int RegisterTable(ServerTable* table);
  // store server data to file
  void StoreTable(int epoch);
  // load data from file and return next iteration number
  int LoadTable(const std::string& file_path);
  void SetTableFilePath(const std::string& table_file_path);
private:
  void ProcessGet(MessagePtr& msg);
  void ProcessAdd(MessagePtr& msg);
  std::string table_file_path_;
  // contains the parameter data structure and related handle method
  std::vector<ServerTable*> store_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_SERVER_H_
