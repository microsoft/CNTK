#ifndef MULTIVERSO_COMMUNICATION_H_
#define MULTIVERSO_COMMUNICATION_H_

#include "multiverso/actor.h"
#include "multiverso/message.h"

namespace multiverso {

class NetInterface;

class Communicator : public Actor {
public:
  Communicator();
  ~Communicator();

private:
  void Main() override;
  // Process message received from other actors, either send to other nodes, or
  // forward to local actors.
  void ProcessMessage(MessagePtr& msg);
  // Thread function to receive messages from other nodes
  void Communicate();
  // Forward to other actors in the same node
  void LocalForward(MessagePtr& msg);

  NetInterface* net_util_;
  std::unique_ptr<std::thread> recv_thread_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_COMMUNICATION_H_
