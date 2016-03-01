#ifndef MULTIVERSO_NET_MPI_NET_H_ 
#define MULTIVERSO_NET_MPI_NET_H_

#ifdef MULTIVERSO_USE_MPI

#include "multiverso/net.h"

#include <limits>
#include <mutex>
#include <queue>

#include "multiverso/message.h"
#include "multiverso/util/log.h"
#include "multiverso/util/mt_queue.h"

#include <mpi.h>


#ifdef _MSC_VER
#undef max
#endif

namespace multiverso {

#define MV_MPI_CALL(mpi_return) CHECK((mpi_return) == MPI_SUCCESS)

class MPINetWrapper : public NetInterface {
public:
  MPINetWrapper() : more_(std::numeric_limits<char>::max()) {}

  class MPIMsgHandle {
  public:
    void add_handle(MPI_Request handle) {
      handles_.push_back(handle);
    }

    void set_msg(MessagePtr& msg) { msg_ = std::move(msg); }
    const MessagePtr& msg() const { return msg_; }
    void set_size(size_t size) { size_ = size; }
    size_t size() const { return size_; }

    void Wait() {
      CHECK_NOTNULL(msg_.get());
      int count = static_cast<int>(handles_.size());
      MPI_Status* status = new MPI_Status[count];
      MV_MPI_CALL(MPI_Waitall(count, handles_.data(), status));
      delete[] status;
    }

    int Test() {
      CHECK_NOTNULL(msg_.get());
      int count = static_cast<int>(handles_.size());
      MPI_Status* status = new MPI_Status[count];
      int flag;
      MV_MPI_CALL(MPI_Testall(count, handles_.data(), &flag, status));
      delete[] status;
      return flag;
    }
  private:
    std::vector<MPI_Request> handles_;
    MessagePtr msg_;
    size_t size_;
  };

  void Init(int* argc, char** argv) override {
    // MPI_Init(argc, &argv);
    MV_MPI_CALL(MPI_Initialized(&inited_));
    if (!inited_) {
      MV_MPI_CALL(MPI_Init_thread(argc, &argv, MPI_THREAD_SERIALIZED, &thread_provided_));
    }
    MV_MPI_CALL(MPI_Query_thread(&thread_provided_));
    if (thread_provided_ < MPI_THREAD_SERIALIZED) {
      Log::Fatal("At least MPI_THREAD_SERIALIZED supported is needed by multiverso.\n");
    }
    else if (thread_provided_ == MPI_THREAD_SERIALIZED) {
      Log::Info("multiverso MPI-Net is initialized under MPI_THREAD_SERIALIZED mode.\n");
    }
    else if (thread_provided_ == MPI_THREAD_MULTIPLE) {
      Log::Debug("multiverso MPI-Net is initialized under MPI_THREAD_MULTIPLE mode.\n");
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
    MPI_Barrier(MPI_COMM_WORLD);
    Log::Debug("%s net util inited, rank = %d, size = %d\n",
      name().c_str(), rank(), size());
  }

  void Finalize() override { MPI_Finalize(); }

  int rank() const override { return rank_; }
  int size() const override { return size_; }
  std::string name() const override { return "MPI"; }

  //size_t Send(MessagePtr& msg) override {
  //  while (!msg_handles_.empty()) {
  //    MPIMsgHandle* prev = msg_handles_.front();
  //    if (prev->Test()) {
  //      delete prev;
  //      prev = nullptr;
  //      msg_handles_.pop();
  //    } else {
  //      break;
  //    }
  //  }
  //  MPIMsgHandle* handle = new MPIMsgHandle();
  //  handle->set_msg(msg);
  //  size_t size = SendAsync(handle->msg(), handle);
  //  handle->set_size(size);
  //  msg_handles_.push(handle);
  //  return size;
  //}

  size_t Send(MessagePtr& msg) override {
    if (msg.get()) { send_queue_.Push(msg); }
    
    if (last_handle_.get() != nullptr && !last_handle_->Test()) {
      // Last msg is still on the air
      return 0;
    }

    // send over, free the last msg
    last_handle_.reset();

    // if there is more msg to send
    if (send_queue_.Empty()) return 0;
    
    // Send a front msg of send queue
    last_handle_.reset(new MPIMsgHandle()); 
    MessagePtr sending_msg;
    CHECK(send_queue_.TryPop(sending_msg));
    last_handle_->set_msg(sending_msg);
    size_t size = SendAsync(last_handle_->msg(), last_handle_.get());
    return size;
  }

  size_t Recv(MessagePtr* msg) override {
    MPI_Status status;
    int flag;
    // non-blocking probe whether message comes
    MV_MPI_CALL(MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status));
    int count;
    MV_MPI_CALL(MPI_Get_count(&status, MPI_BYTE, &count));
    if (!flag) return 0;
    CHECK(count == Message::kHeaderSize);
    return RecvMsgFrom(status.MPI_SOURCE, msg);
  }

  int thread_level_support() override { 
    if (thread_provided_ == MPI_THREAD_MULTIPLE) 
      return NetThreadLevel::THREAD_MULTIPLE;
    return NetThreadLevel::THREAD_SERIALIZED; 
  }

private:
  size_t SendAsync(const MessagePtr& msg, 
                   MPIMsgHandle* msg_handle) {
    CHECK_NOTNULL(msg_handle);
    size_t size = Message::kHeaderSize;
    MPI_Request handle;
    CHECK_NOTNULL(msg->header());
    MV_MPI_CALL(MPI_Isend(msg->header(), Message::kHeaderSize, MPI_BYTE,
      msg->dst(), 0, MPI_COMM_WORLD, &handle));
    msg_handle->add_handle(handle);
    // Send multiple msg 
    for (auto& blob : msg->data()) {
      CHECK_NOTNULL(blob.data());
      MV_MPI_CALL(MPI_Isend(blob.data(), static_cast<int>(blob.size()), 
        MPI_BYTE, msg->dst(),
        0, MPI_COMM_WORLD, &handle));
      size += blob.size();
      msg_handle->add_handle(handle);
    }
    // Send an extra over tag indicating the finish of this Message
    MV_MPI_CALL(MPI_Isend(&more_, sizeof(char), MPI_BYTE, msg->dst(),
      0, MPI_COMM_WORLD, &handle));
    // Log::Debug("MPI-Net: rank %d send msg size = %d\n", rank(), size+4);
    msg_handle->add_handle(handle);
    return size + sizeof(char);
  }

  size_t RecvMsgFrom(int source, MessagePtr* msg_ptr) {
    if (!msg_ptr->get()) msg_ptr->reset(new Message());
    MessagePtr& msg = *msg_ptr;
    msg->data().clear();
    MPI_Status status;
    CHECK_NOTNULL(msg->header());
    MV_MPI_CALL(MPI_Recv(msg->header(), Message::kHeaderSize,
      MPI_BYTE, source, 0, MPI_COMM_WORLD, &status));
    size_t size = Message::kHeaderSize;
    while (true) {
      int count;
      MV_MPI_CALL(MPI_Probe(source, 0, MPI_COMM_WORLD, &status));
      MV_MPI_CALL(MPI_Get_count(&status, MPI_BYTE, &count));
      Blob blob(count);
      // We only receive from msg->src() until we recv the overtag msg
      MV_MPI_CALL(MPI_Recv(blob.data(), count, MPI_BYTE, source,
        0, MPI_COMM_WORLD, &status));
      size += count;
      if (count == sizeof(char)) {
        if (blob.As<char>() == more_) break;
        CHECK(false);
      }
      msg->Push(blob);
    }
    return size;
  }

private:
  const char more_;
  std::mutex mutex_;
  int thread_provided_;
  int inited_;
  int rank_;
  int size_;
  // std::queue<MPIMsgHandle *> msg_handles_;
  std::unique_ptr<MPIMsgHandle> last_handle_;
  MtQueue<MessagePtr> send_queue_;
};

}

#endif // MULTIVERSO_USE_MPI

#endif // MULTIVERSO_NET_MPI_NET_H_