#ifndef MULTIVERSO_NET_NET_H_
#define MULTIVERSO_NET_NET_H_

#include <string>
#include "multiverso/message.h"

namespace multiverso {

enum NetThreadLevel {
  THREAD_SERIALIZED,
  THREAD_MULTIPLE
};

// Interface of inter process communication method
class NetInterface {
public:
  static NetInterface* Get();

  virtual void Init(int* argc = nullptr, char** argv = nullptr) = 0;

  virtual void Finalize() = 0;

  // Bind with a specific endpoint
  virtual int  Bind(int rank, char* endpoint) = 0;
  // Connect with other endpoints
  virtual int  Connect(int* rank, char* endpoints[], int size) = 0;

  virtual bool active() const = 0;

  virtual std::string name() const = 0;
  virtual int size() const = 0;
  virtual int rank() const = 0;

  // \return 1. > 0 sent size 2. = 0 not sent 3. < 0 net error
  virtual int Send(MessagePtr& msg) = 0;

  // \return 1. > 0 received size 2. = 0 not received 3. < 0 net error
  virtual int Recv(MessagePtr* msg) = 0;

  // Blocking, send raw data to rank
  virtual void SendTo(int rank, char* buf, int len) const = 0;
  // Blocking, receive raw data from rank 
  virtual void RecvFrom(int rank, char* buf, int len) const = 0;
  // Blocking, send and recv at same time
  virtual void SendRecv(int send_rank, char* send_buf, int send_len,
    int recv_rank, char* recv_buf, int recv_len) const = 0;

  virtual int thread_level_support() = 0;
};

namespace net {

// inplace allreduce
template <typename Typename>
void Allreduce(Typename* data, size_t elem_count);

}

}  // namespace multiverso

#endif  // MULTIVERSO_NET_NET_H_
