#ifndef MULTIVERSO_CONTROLLER_H_
#define MULTIVERSO_CONTROLLER_H_

#include "multiverso/actor.h"
#include "multiverso/message.h"

namespace multiverso {

class Controller : public Actor {
public:
  Controller();
  ~Controller();

private:
  void ProcessBarrier(MessagePtr& msg);
  void ProcessRegister(MessagePtr& msg);

  class RegisterController;
  RegisterController* register_controller_;
  class BarrierController;
  BarrierController* barrier_controller_;
};

}  // namespace multiverso

#endif  // MULTIVERSO_CONTROLLER_H_
