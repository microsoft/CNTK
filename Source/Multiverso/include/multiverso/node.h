#ifndef MULTIVERSO_NODE_H_
#define MULTIVERSO_NODE_H_

namespace multiverso {

enum Role {
  WORKER = 1,
  SERVER = 2
};

struct Node {
  int rank;
  // role can be 0, 1, 2, 3
  // 00 means neither worker nor server, should be controllor, so at most
  // one node could use this value
  // 01 means worker
  // 10 means server
  // 11 means both server and worker, default value
  int role;
  // bool is_controller; // currently rank == 0 means controller
  int worker_id;
  int server_id;

  Node();
};

namespace node {

bool is_worker(int role);
bool is_server(int role);

}  // namespace node

}  // namespace multiverso

#endif  // MULTIVERSO_NODE_H_
