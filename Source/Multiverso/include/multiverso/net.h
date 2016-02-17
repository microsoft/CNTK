#ifndef MULTIVERSO_NET_NET_H_
#define MULTIVERSO_NET_NET_H_


#include <string>
#include "multiverso/message.h"

namespace multiverso {
// Interface of inter process communication method
class NetInterface {
public:
  static NetInterface* Get();
  virtual void Init(int* argc = nullptr, char** argv = nullptr) = 0;
  virtual void Finalize() = 0;

  virtual std::string name() const = 0;
  virtual int size() const = 0;
  virtual int rank() const = 0;

  virtual size_t Send(const MessagePtr& msg) = 0;
  virtual size_t Recv(MessagePtr* msg) = 0;
};

}
#endif // MULTIVERSO_NET_NET_H_