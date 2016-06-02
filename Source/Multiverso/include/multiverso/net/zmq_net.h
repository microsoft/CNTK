#ifndef MULTIVERSO_NET_ZMQ_NET_H_
#define MULTIVERSO_NET_ZMQ_NET_H_

#ifdef MULTIVERSO_USE_ZMQ

#include "multiverso/net.h"

#include <limits>
#include <thread>

#include "multiverso/message.h"
#include "multiverso/util/log.h"
#include "multiverso/util/net_util.h"

#include <zmq.h>

namespace multiverso {
class ZMQNetWrapper : public NetInterface {
public:
  // argc >= 2
  // argv[1]: machine file, format is same with MPI machine file
  // argv[2]: port used
  void Init(int* argc, char** argv) override {
    // get machine file 
    if (active_) return;
    CHECK(*argc > 2);
    std::vector<std::string> machine_lists;
    ParseMachineFile(argv[1], &machine_lists);
    int port = atoi(argv[2]);

    size_ = static_cast<int>(machine_lists.size());
    CHECK(size_ > 0);
    std::unordered_set<std::string> local_ip;
    net::GetLocalIPAddress(&local_ip);

    context_ = zmq_ctx_new();
    zmq_ctx_set(context_, ZMQ_MAX_SOCKETS, 256);

    for (auto ip : machine_lists) {
      if (local_ip.find(ip) != local_ip.end()) { // my rank
        rank_ = static_cast<int>(senders_.size());
        senders_.push_back(nullptr);
        receiver_ = zmq_socket(context_, ZMQ_DEALER);
        int rc = zmq_bind(receiver_, 
          ("tcp://" + ip + ":" + std::to_string(port)).c_str());
        CHECK(rc == 0);
      } else {
        void* senders = zmq_socket(context_, ZMQ_DEALER);
        int rc = zmq_connect(senders, 
          ("tcp://" + ip + ":" + std::to_string(port)).c_str());
        CHECK(rc == 0);
        senders_.push_back(senders);
      }
    }
    CHECK_NOTNULL(receiver_);
    active_ = true;
    Log::Info("%s net util inited, rank = %d, size = %d\n",
      name().c_str(), rank(), size());
  }

  virtual int Bind(int rank, char* endpoint) override {
    rank_ = rank;
    std::string ip_port(endpoint);
    if (context_ == nullptr) { context_ = zmq_ctx_new(); }
    CHECK_NOTNULL(context_);
    receiver_ = zmq_socket(context_, ZMQ_DEALER);
    int rc = zmq_bind(receiver_, ("tcp://" + ip_port).c_str());
    if (rc == 0) {
      return 0;
    }
    else {
      Log::Error("Failed to bind the socket for receiver, ip:port = %s\n", 
                 endpoint);
      return -1;
    }
  }

  // Connect with other endpoints
  virtual int Connect(int* ranks, char* endpoints[], int size) override {
    CHECK_NOTNULL(receiver_);
    CHECK_NOTNULL(context_);
    size_ = size;
    senders_.resize(size_);
    for (int i = 0; i < size; ++i) {
      int rank = ranks[i];
      if (rank == rank_) continue;
      std::string ip_port(endpoints[i]);
      senders_[rank] = zmq_socket(context_, ZMQ_DEALER);
      int rc = zmq_connect(senders_[rank], ("tcp://" + ip_port).c_str());
      if (rc != 0) {
        Log::Error("Failed to connect the socket for sender, rank = %d, " 
                    "ip:port = %s\n", rank, endpoints[i]);
        return -1;
      }
    }
    active_ = true;
    return 0;
  }

  void Finalize() override {
    active_ = false;
    // zmq_term(context_);
    zmq_close(receiver_);
    for (auto& p : senders_) if (p) zmq_close(p);
    zmq_ctx_destroy(context_);
  }

  bool active() const override { return active_; }
  int rank() const override { return rank_; }
  int size() const override { return size_; }
  std::string name() const override { return "ZeroMQ"; }

  size_t Send(MessagePtr& msg) override {
    size_t size = 0;
    int dst = msg->dst();
    void* socket = senders_[dst];
    CHECK_NOTNULL(socket);
    int send_size;
    send_size = zmq_send(socket, msg->header(), 
      Message::kHeaderSize, msg->data().size() > 0 ? ZMQ_SNDMORE : 0);
    CHECK(Message::kHeaderSize == send_size);
    size += send_size;
    for (size_t i = 0; i < msg->data().size(); ++i) {
      Blob blob = msg->data()[i];
      size_t blob_size = blob.size();
      CHECK_NOTNULL(blob.data());
      send_size = zmq_send(socket, &blob_size, sizeof(size_t), ZMQ_SNDMORE);
      CHECK(send_size == sizeof(size_t));
      send_size = zmq_send(socket, blob.data(), static_cast<int>(blob.size()),
        i == msg->data().size() - 1 ? 0 : ZMQ_SNDMORE);
      CHECK(send_size == blob_size);
      size += blob_size + sizeof(size_t);
    }
    return size;
  }

  size_t Recv(MessagePtr* msg_ptr) override {
    if (!msg_ptr->get()) msg_ptr->reset(new Message());
    size_t size = 0;
    int recv_size;
    size_t blob_size;
    int more;
    size_t more_size = sizeof(more);
    // Receiving a Message from multiple zmq_recv
    CHECK_NOTNULL(msg_ptr);
    MessagePtr& msg = *msg_ptr;
    msg->data().clear();
    CHECK(msg.get());
    recv_size = zmq_recv(receiver_, msg->header(), Message::kHeaderSize, 0);
    if (recv_size < 0) { return -1; }
    CHECK(Message::kHeaderSize == recv_size);

    size += recv_size;
    zmq_getsockopt(receiver_, ZMQ_RCVMORE, &more, &more_size);

    while (more) {
      recv_size = zmq_recv(receiver_, &blob_size, sizeof(size_t), 0);
      CHECK(recv_size == sizeof(size_t));
      size += recv_size;
      zmq_getsockopt(receiver_, ZMQ_RCVMORE, &more, &more_size);
      CHECK(more);
      Blob blob(blob_size);
      recv_size = zmq_recv(receiver_, blob.data(), blob.size(), 0);
      CHECK(recv_size == blob_size);
      size += recv_size;
      msg->Push(blob);
      zmq_getsockopt(receiver_, ZMQ_RCVMORE, &more, &more_size);
    }
    return size;
  }


  void SendTo(int rank, char* buf, int len) const override {
    int send_size = 0;
    while (send_size < len) {
      int cur_size = zmq_send(senders_[rank], buf + send_size, len - send_size, 0);
      if (cur_size < 0) { Log::Error("socket send error %d", cur_size); }
      send_size += cur_size;
    }
  }

  void RecvFrom(int, char* buf, int len) const override {
    // note: rank is not used here
    int recv_size = 0;
    while (recv_size < len) {
      int cur_size = zmq_recv(receiver_, buf + recv_size, len - recv_size, 0);
      if (cur_size < 0) { Log::Error("socket receive error %d", cur_size); }
      recv_size += cur_size;
    }
  }

  void SendRecv(int send_rank, char* send_buf, int send_len,
    int recv_rank, char* recv_buf, int recv_len) const override {
    // send first
    SendTo(send_rank, send_buf, send_len);
    // then recv
    RecvFrom(recv_rank, recv_buf, recv_len);
    // wait for send complete
  }

  int thread_level_support() override { 
    return NetThreadLevel::THREAD_MULTIPLE; 
  }

protected:
  void ParseMachineFile(std::string filename, 
                        std::vector<std::string>* result) {
    CHECK_NOTNULL(result);
    FILE* file;
    char str[32];
    int i = 0;
#ifdef _MSC_VER
    fopen_s(&file, filename.c_str(), "r");
#else
    file = fopen(filename.c_str(), "r");
#endif
    CHECK_NOTNULL(file);
#ifdef _MSC_VER
    while (fscanf_s(file, "%s", &str, 32) > 0) {
#else
    while (fscanf(file, "%s", &str) > 0) {
#endif
      result->push_back(str);
    }
    fclose(file);
  }

  bool active_;
  void* context_;
  void* receiver_;
  std::vector<void*> senders_;
  int rank_;
  int size_;
};
} // namespace multiverso

#endif // MULTIVERSO_USE_ZEROMQ

#endif // MULTIVERSO_NET_ZMQ_NET_H_