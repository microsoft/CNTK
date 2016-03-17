#ifndef MULTIVERSO_ZOO_H_
#define MULTIVERSO_ZOO_H_

#include <atomic>
#include <string>
#include <unordered_map>

#include "multiverso/actor.h"
#include "multiverso/node.h"
#include "multiverso/table_interface.h"

namespace multiverso {

class NetInterface;

// The dashboard 
// 1. Manage all components in the system, include all actors, and network env
// 2. Maintain system information, provide method to access this information
// 3. Control the system, to start and end
class Zoo {
public:
  ~Zoo();
  static Zoo* Get() { static Zoo zoo; return &zoo; };

  // Start all actors
  void Start(int* argc, char** argv, int role);
  // Stop all actors
  void Stop(bool finalize_net);

  void Barrier();

  void SendTo(const std::string& name, MessagePtr&);
  void Receive(MessagePtr& msg);

  int rank() const;
  int size() const;

  // TODO(to change)
  int worker_rank() const { return nodes_[rank()].worker_id; }
  int server_rank() const { return nodes_[rank()].server_id; }

  int worker_id_to_rank(int worker_id) const { 
    return worker_id_to_rank_[worker_id]; 
  }

  int server_id_to_rank(int server_id) const {
    return server_id_to_rank_[server_id];
  }

  int num_workers() const { return num_workers_; }
  int num_servers() const { return num_servers_; }


  int RegisterTable(WorkerTable* worker_table);
  int RegisterTable(ServerTable* server_table);

  void RegisterActor(const std::string name, Actor* actor) {
    CHECK(zoo_[name] == nullptr);
    zoo_[name] = actor;
  }
private:
  // private constructor
  Zoo();
  void RegisterNode();

  std::unordered_map<std::string, Actor*> zoo_;

  std::unique_ptr<MtQueue<MessagePtr>> mailbox_;

  NetInterface* net_util_;

  std::vector<Node> nodes_;
  std::vector<int> server_id_to_rank_;
  std::vector<int> worker_id_to_rank_;

  int num_workers_;
  int num_servers_;
};

}

#endif // MULTIVERSO_ZOO_H_
