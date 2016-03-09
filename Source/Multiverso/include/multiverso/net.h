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

  // \return 1. > 0 sended size
  //         2. = 0 not sended
  //         3. < 0 net error
  virtual size_t Send(MessagePtr& msg) = 0;

  // \return 1. > 0 received size
  //         2. = 0 not received
  //         3. < 0 net error
  virtual size_t Recv(MessagePtr* msg) = 0;

  virtual int thread_level_support() = 0;
};

}
#endif // MULTIVERSO_NET_NET_H_