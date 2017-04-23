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
  static Server* GetServer();
  int RegisterTable(ServerTable* table);

protected:
  virtual void ProcessGet(MessagePtr& msg);
  virtual void ProcessAdd(MessagePtr& msg);

  std::vector<ServerTable*> store_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_SERVER_H_
