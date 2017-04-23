#include "multiverso/zoo.h"

#include <string>

#include "multiverso/actor.h"
#include "multiverso/communicator.h"
#include "multiverso/controller.h"
#include "multiverso/dashboard.h"
#include "multiverso/message.h"
#include "multiverso/net.h"
#include "multiverso/server.h"
#include "multiverso/util/configure.h"
#include "multiverso/util/log.h"
#include "multiverso/util/mt_queue.h"
#include "multiverso/worker.h"

namespace multiverso {

Zoo::Zoo() {}

Zoo::~Zoo() {}

MV_DEFINE_string(ps_role, "default", "none / worker / server / default");
MV_DEFINE_bool(ma, false, "model average, will not start server if true");
MV_DECLARE_bool(sync);

namespace {

int ParsePSRole(const std::string& ps_role) {
  if (ps_role == "none")    return Role::NONE;
  if (ps_role == "worker")  return Role::WORKER; 
  if (ps_role == "server")  return Role::SERVER;
  if (ps_role == "default") return Role::ALL;
  return -1;
}

const int kController = 0;

}  // namespace

void Zoo::Start(int* argc, char** argv) {
  Log::Debug("Zoo started\n");
  ParseCMDFlags(argc, argv);

  // Init the network
  net_util_ = NetInterface::Get();
  net_util_->Init(argc, argv);

  if (!MV_CONFIG_ma) { StartPS(); }
}

void Zoo::Stop(bool finalize_net) {
  // Stop the system
  if (!MV_CONFIG_ma) { StopPS(); }
  // Stop the network
  if (finalize_net) net_util_->Finalize();
  for (auto actor : zoo_) delete actor.second;
  zoo_.clear();
  Log::Info("Multiverso Shutdown successfully\n");
}

int Zoo::rank() const { return NetInterface::Get()->rank(); }
int Zoo::size() const { return NetInterface::Get()->size(); }

void Zoo::SendTo(const std::string& name, MessagePtr& msg) {
  CHECK(zoo_.find(name) != zoo_.end());
  zoo_[name]->Receive(msg);
}
void Zoo::Receive(MessagePtr& msg) {
  mailbox_->Push(msg);
}

void Zoo::StartPS() {
  int role = ParsePSRole(MV_CONFIG_ps_role);
  CHECK(role != -1);

  nodes_.resize(size());
  nodes_[rank()].rank = rank();
  nodes_[rank()].role = role;
  mailbox_.reset(new MtQueue<MessagePtr>);

  // NOTE(feiga): the start order is non-trivial, communicator should be last.
  if (rank() == kController) { 
    Actor* controler = new Controller(); 
    controler->Start(); 
  }
  Actor* communicator = new Communicator();
  communicator->Start();
  // activate the system
  RegisterNode();

  if (node::is_server(role)) {
    Actor* server = Server::GetServer();
    server->Start();
  }
  if (node::is_worker(role)) {
    Actor* worker = new Worker();
    worker->Start();
  }
  Barrier();
  Log::Info("Rank %d: Multiverso start successfully\n", rank());
}

void Zoo::StopPS() {
  if (MV_CONFIG_sync) {
    FinishTrain();
  }
  Barrier();

  // Stop all actors
  for (auto actor : zoo_) { 
    actor.second->Stop(); 
  }
}

void Zoo::RegisterNode() {
  MessagePtr msg(new Message());
  msg->set_src(rank());
  msg->set_dst(kController);
  msg->set_type(MsgType::Control_Register);
  msg->Push(Blob(&nodes_[rank()], sizeof(Node)));
  SendTo(actor::kCommunicator, msg);

  // waif for reply
  mailbox_->Pop(msg);
  CHECK(msg->type() == MsgType::Control_Reply_Register);
  CHECK(msg->data().size() == 2);
  Blob info_blob = msg->data()[0];
  Blob count_blob = msg->data()[1];
  num_workers_ = count_blob.As<int>(0);
  num_servers_ = count_blob.As<int>(1);
  worker_id_to_rank_.resize(num_workers_);
  server_id_to_rank_.resize(num_servers_);
  CHECK(info_blob.size() == size() * sizeof(Node));
  memcpy(nodes_.data(), info_blob.data(), info_blob.size());
  for (auto node : nodes_) {
    if (node.worker_id != -1) {
      worker_id_to_rank_[node.worker_id] = node.rank;
    }
    if (node.server_id != -1) {
      server_id_to_rank_[node.server_id] = node.rank;
    }
  }
  Log::Debug("rank %d end register\n", Zoo::Get()->rank());
}

void Zoo::RegisterActor(const std::string name, Actor* actor) {
  CHECK(zoo_[name] == nullptr);
  zoo_[name] = actor;
}

void Zoo::FinishTrain() {
  for (auto i = 0; i < num_servers_; i++) {
    int dst_rank = server_id_to_rank(i);
    MessagePtr msg(new Message());
    msg->set_src(rank());
    msg->set_dst(dst_rank);
    msg->set_type(MsgType::Server_Finish_Train);
    SendTo(actor::kCommunicator, msg);
  }
}


void Zoo::Barrier() {
  MessagePtr msg(new Message());
  msg->set_src(rank());
  msg->set_dst(kController); 
  msg->set_type(MsgType::Control_Barrier);
  SendTo(actor::kCommunicator, msg);

  Log::Debug("rank %d requested barrier.\n", rank());
  // wait for reply
  mailbox_->Pop(msg);
  CHECK(msg->type() == MsgType::Control_Reply_Barrier);
  Log::Debug("rank %d reached barrier\n", rank());
}

int Zoo::RegisterTable(WorkerTable* worker_table) {
  return dynamic_cast<Worker*>(zoo_[actor::kWorker])
    ->RegisterTable(worker_table);
}

int Zoo::RegisterTable(ServerTable* server_table) {
  return dynamic_cast<Server*>(zoo_[actor::kServer])
    ->RegisterTable(server_table);
}

}  // namespace multiverso
