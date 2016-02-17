#ifndef MULTIVERSO_MESSAGE_H_
#define MULTIVERSO_MESSAGE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "multiverso/blob.h" 

namespace multiverso {

enum MsgType {
  Request_Get = 1,
  Request_Add = 2,
  Reply_Get = -1,
  Reply_Add = -2,
  Control_Barrier = 33, // 0x100001
  Control_Reply_Barrier = -33,
  Control_Register = 34,
  Control_Reply_Register = -34,
  Default = 0
};

class Message {
public:
  Message() { }

  ~Message() { }

  MsgType type() const { return static_cast<MsgType>(header_[2]); }
  int src() const { return header_[0]; }
  int dst() const { return header_[1]; }
  int table_id() const { return header_[3]; }
  int msg_id() const { return header_[4]; }

  void set_type(MsgType type) { header_[2] = static_cast<int>(type); }
  void set_src(int src) { header_[0] = src; }
  void set_dst(int dst) { header_[1] = dst; }
  void set_table_id(int table_id) { header_[3] = table_id; }
  void set_msg_id(int msg_id) { header_[4] = msg_id; }

  void set_data(const std::vector<Blob>& data) { data_ = std::move(data); }
  std::vector<Blob>& data() { return data_; }
  size_t size() const { return data_.size(); }

  int* header() { return header_; }
  const int* header() const { return header_; }
  static const int kHeaderSize = 8 * sizeof(int);

  // Deep Copy
  Message* CopyFrom(const Message& src);
  // Create a Message with only headers
  // The src/dst, type is opposite with src message
  Message* CreateReplyMessage() {
    Message* reply = new Message();
    reply->set_dst(this->src());
    reply->set_src(this->dst());
    reply->set_type(static_cast<MsgType>(-header_[2]));
    reply->set_table_id(this->table_id());
    reply->set_msg_id(this->msg_id());
    return reply;
  }

  void Push(const Blob& blob) { data_.push_back(blob); }

private:
  int header_[8];
  std::vector<Blob> data_;
};

typedef std::unique_ptr<Message> MessagePtr;

}

#endif // MULTIVERSO_MESSAGE_H_
