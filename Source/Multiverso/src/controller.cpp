#include "multiverso/controller.h"

#include <vector>

#include "multiverso/message.h"
#include "multiverso/node.h"
#include "multiverso/util/log.h"
#include "multiverso/zoo.h"

namespace multiverso {

class Controller::BarrierController {
public:
  explicit BarrierController(Controller* parent) : parent_(parent) {}

  void Control(MessagePtr& msg) {
    tasks_.push_back(std::move(msg));
    if (static_cast<int>(tasks_.size()) == Zoo::Get()->size()) {
      MessagePtr my_reply;  // my reply should be the last one
      for (auto& task_msg : tasks_) {
        MessagePtr reply(task_msg->CreateReplyMessage());
        if (reply->dst() != Zoo::Get()->rank()) {
          parent_->SendTo(actor::kCommunicator, reply);
        } else {
          my_reply = std::move(reply);
        }
      }
      parent_->SendTo(actor::kCommunicator, my_reply);
      tasks_.clear();
    }
  }

private:
  std::vector<MessagePtr> tasks_;
  Controller* parent_;  // not owned
};

class Controller::RegisterController {
public:
  explicit RegisterController(Controller* parent) :
    num_registered_(0), num_server_(0), num_worker_(0),
    parent_(parent) {
    all_nodes_.resize(Zoo::Get()->size());
  }

  void Control(MessagePtr& msg) {
    int src = msg->src();
    CHECK(msg->size() == 1);
    CHECK(src < static_cast<int>(all_nodes_.size()) && src >= 0);
    all_nodes_[src] = *(reinterpret_cast<Node*>(msg->data()[0].data()));
    if (node::is_worker(all_nodes_[src].role))
      all_nodes_[src].worker_id = num_worker_++;
    if (node::is_server(all_nodes_[src].role))
      all_nodes_[src].server_id = num_server_++;
    if (++num_registered_ == Zoo::Get()->size()) {  // all nodes is registered
      Log::Info("All nodes registered. System contains %d nodes. num_worker = "
        "%d, num_server = %d\n", Zoo::Get()->size(), num_worker_, num_server_);
      Blob info_blob(all_nodes_.data(), all_nodes_.size() * sizeof(Node));
      Blob count_blob(2 * sizeof(int));
      count_blob.As<int>(0) = num_worker_;
      count_blob.As<int>(1) = num_server_;
      for (int i = Zoo::Get()->size() - 1; i >= 0; --i) {  // let rank 0 be last
        MessagePtr reply(new Message());
        reply->set_src(Zoo::Get()->rank());
        reply->set_dst(i);
        reply->set_type(MsgType::Control_Reply_Register);
        reply->Push(info_blob);
        reply->Push(count_blob);
        parent_->SendTo(actor::kCommunicator, reply);
      }
    }
  }

private:
  int num_registered_;
  int num_server_;
  int num_worker_;
  std::vector<Node> all_nodes_;
  Controller* parent_;  // not owned
};

Controller::Controller() : Actor(actor::kController) {
  RegisterHandler(MsgType::Control_Barrier, std::bind(
    &Controller::ProcessBarrier, this, std::placeholders::_1));
  RegisterHandler(MsgType::Control_Register, std::bind(
    &Controller::ProcessRegister, this, std::placeholders::_1));
  barrier_controller_ = new BarrierController(this);
  register_controller_ = new RegisterController(this);
}

Controller::~Controller() {
  delete barrier_controller_;
  delete register_controller_;
}

void Controller::ProcessBarrier(MessagePtr& msg) {
  barrier_controller_->Control(msg);
}

void Controller::ProcessRegister(MessagePtr& msg) {
  register_controller_->Control(msg);
}

}  // namespace multiverso
