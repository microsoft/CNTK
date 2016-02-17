#ifndef MULTIVERSO_SERVER_H_
#define MULTIVERSO_SERVER_H_

#include "multiverso/actor.h"
#include <vector>

namespace multiverso {

class ServerTable;

class Server : public Actor { 
public:
  Server();
  int RegisterTable(ServerTable* table);
private:
  void ProcessGet(MessagePtr& msg);
  void ProcessAdd(MessagePtr& msg);
  // contains the parameter data structure and related handle method
  std::vector<ServerTable*> store_;
};

}
#endif // MULTIVERSO_SERVER_H_