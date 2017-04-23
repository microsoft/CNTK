#include "multiverso/actor.h"

#include <chrono>
#include <string>
#include <thread>

#include "multiverso/message.h"
#include "multiverso/util/log.h"
#include "multiverso/util/mt_queue.h"
#include "multiverso/zoo.h"

namespace multiverso {

Actor::Actor(const std::string& name) : name_(name) {
  mailbox_.reset(new MtQueue<MessagePtr>());
  Zoo::Get()->RegisterActor(name, this);
  is_working_ = false;
}

Actor::~Actor() {}

void Actor::Start() {
  thread_.reset(new std::thread(&Actor::Main, this));
  while (!is_working_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void Actor::Stop() {
  while (!mailbox_->Empty()) { ; }
  is_working_ = false;
  mailbox_->Exit();
  thread_->join();
}

void Actor::Receive(MessagePtr& msg) { mailbox_->Push(msg); }

void Actor::Main() {
  is_working_ = true;
  MessagePtr msg;
  while (mailbox_->Pop(msg)) {
    if (handlers_.find(msg->type()) != handlers_.end()) {
      handlers_[msg->type()](msg);
    } else if (handlers_.find(MsgType::Default) != handlers_.end()) {
      handlers_[MsgType::Default](msg);
    } else {
      Log::Fatal("Unexpected msg type\n");
    }
  }
}

void Actor::SendTo(const std::string& dst_name, MessagePtr& msg) {
  Zoo::Get()->SendTo(dst_name, msg);
}

}  // namespace multiverso
