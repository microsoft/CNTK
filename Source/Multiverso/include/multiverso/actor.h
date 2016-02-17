#ifndef MULTIVERSO_ACTOR_H_
#define MULTIVERSO_ACTOR_H_

#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "multiverso/message.h"

namespace multiverso {

template<typename T> class MtQueue;

// The basic computation and communication unit in the system 
class Actor {
public:
  explicit Actor(const std::string& name);
  virtual ~Actor();
  // Start to run the Actor
  void Start();
  // Stop to run the Actor
  void Stop();
  // Accept a message from other actors
  void Accept(MessagePtr&);

  const std::string name() const { return name_; }

  // Message response function
  using Task = std::function<void(MessagePtr&)>;

protected:

  void RegisterTask(const MsgType& type, const Task& task) {
    handlers_.insert({ type, task });
  }
  void DeliverTo(const std::string& dst_name, MessagePtr& msg);

  // Run in a background thread to receive msg from other actors and process
  // messages based on registered handlers
  virtual void Main();

  std::string name_;

  std::unique_ptr<std::thread> thread_;
  // message queue
  std::unique_ptr<MtQueue<std::unique_ptr<Message>> > mailbox_;
  std::unordered_map<int, Task> handlers_;
};

namespace actor {

  const std::string kCommunicator = "communicator";
  const std::string kController = "controller";
  const std::string kServer = "server";
  const std::string kWorker = "worker";

}

} // namespace multiverso

#endif // MULTIVERSO_ANIMAL_H_
