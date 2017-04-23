#include "multiverso/table_interface.h"

#include <mutex>

#include "multiverso/dashboard.h"
#include "multiverso/updater/updater.h"
#include "multiverso/util/log.h"
#include "multiverso/util/waiter.h"
#include "multiverso/zoo.h"

namespace multiverso {

WorkerTable::WorkerTable() {
  msg_id_ = 0;
  m_ = new std::mutex();
  table_id_ = Zoo::Get()->RegisterTable(this);
}

WorkerTable::~WorkerTable() {
  delete m_;
}

ServerTable::ServerTable() {
  Zoo::Get()->RegisterTable(this);
}

void WorkerTable::Get(Blob keys, 
                      const GetOption* option) {
  MONITOR_BEGIN(WORKER_TABLE_SYNC_GET)
  Wait(GetAsync(keys, option));
  MONITOR_END(WORKER_TABLE_SYNC_GET)
}

void WorkerTable::Add(Blob keys, Blob values,
                      const AddOption* option) {
  MONITOR_BEGIN(WORKER_TABLE_SYNC_ADD)
  Wait(AddAsync(keys, values, option));
  MONITOR_END(WORKER_TABLE_SYNC_ADD)
}

int WorkerTable::GetAsync(Blob keys,
                          const GetOption* option) {
  m_->lock();
  int id = msg_id_++;
  waitings_.push_back(new Waiter());
  m_->unlock();
  MessagePtr msg(new Message());
  msg->set_src(Zoo::Get()->rank());
  msg->set_type(MsgType::Request_Get);
  msg->set_msg_id(id);
  msg->set_table_id(table_id_);
  msg->Push(keys);
  // Add general option if necessary
  if (option != nullptr) {
    Blob general_option(option->data(), option->size());
    msg->Push(general_option);
  }
  Zoo::Get()->SendTo(actor::kWorker, msg);
  return id;
}

int WorkerTable::AddAsync(Blob keys, Blob values,
                          const AddOption* option) {
  m_->lock();
  int id = msg_id_++;
  waitings_.push_back(new Waiter());
  m_->unlock();
  MessagePtr msg(new Message());
  msg->set_src(Zoo::Get()->rank());
  msg->set_type(MsgType::Request_Add);
  msg->set_msg_id(id);
  msg->set_table_id(table_id_);
  msg->Push(keys);
  msg->Push(values);
  // Add update option if necessary
  if (option != nullptr) {
    Blob update_option(option->data(), option->size());
    msg->Push(update_option);
  }
  Zoo::Get()->SendTo(actor::kWorker, msg);
  return id;
}

void WorkerTable::Wait(int id) {
  // CHECK(waitings_.find(id) != waitings_.end());
  m_->lock();
  CHECK(waitings_[id] != nullptr);
  Waiter* w = waitings_[id];
  m_->unlock();

  w->Wait();

  m_->lock();
  delete waitings_[id];
  waitings_[id] = nullptr;
  m_->unlock();
}

void WorkerTable::Reset(int msg_id, int num_wait) {
  m_->lock();
  CHECK_NOTNULL(waitings_[msg_id]);
  waitings_[msg_id]->Reset(num_wait);
  m_->unlock();
}

void WorkerTable::Notify(int id) {
  m_->lock();
  CHECK_NOTNULL(waitings_[id]);
  waitings_[id]->Notify();
  m_->unlock();
}

}  // namespace multiverso
