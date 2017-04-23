#ifndef MULTIVERSO_WORKER_H_
#define MULTIVERSO_WORKER_H_

#include <vector>

#include "multiverso/actor.h"

namespace multiverso {

class WorkerTable;

class Worker : public Actor {
public:
  Worker();

  int RegisterTable(WorkerTable* worker_table);

private:
  void ProcessGet(MessagePtr& msg);
  void ProcessAdd(MessagePtr& msg);
  void ProcessReplyGet(MessagePtr& msg);
  void ProcessReplyAdd(MessagePtr& msg);

  std::vector<WorkerTable*> cache_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_WORKER_H_
