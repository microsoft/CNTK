#ifndef MULTIVERSO_NODE_H_
#define MULTIVERSO_NODE_H_

namespace multiverso {

enum Role {
  NONE   = 0,
  WORKER = 1,
  SERVER = 2,
  ALL    = 3
};

struct Node {
  int rank;
  int role;
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
