#include <boost/test/unit_test.hpp>
#include <multiverso/message.h>

namespace multiverso {
namespace test {

BOOST_AUTO_TEST_SUITE(message)

BOOST_AUTO_TEST_CASE(message_access) {
  multiverso::Message msg;
  BOOST_CHECK_EQUAL(msg.data().size(), 0);

  msg.set_msg_id(0);
  BOOST_CHECK_EQUAL(msg.msg_id(), 0);
  msg.set_src(1);
  BOOST_CHECK_EQUAL(msg.src(), 1);
  msg.set_dst(2);
  BOOST_CHECK_EQUAL(msg.dst(), 2);
  msg.set_table_id(3);
  BOOST_CHECK_EQUAL(msg.table_id(), 3);
  msg.set_type(MsgType::Request_Get);
  BOOST_CHECK_EQUAL(msg.type(), MsgType::Request_Get);

  BOOST_TEST_MESSAGE("before blob\n");

  multiverso::Blob data;
  msg.Push(data);
  BOOST_CHECK_EQUAL(msg.size(), 1);


  std::vector<multiverso::Blob> vec_data;
  msg.set_data(vec_data);

  BOOST_CHECK_EQUAL(msg.size(), 0);

  MessagePtr reply_msg(msg.CreateReplyMessage());
  BOOST_CHECK_EQUAL(reply_msg->src(), msg.dst());
  BOOST_CHECK_EQUAL(reply_msg->dst(), msg.src());
  BOOST_CHECK_EQUAL(reply_msg->type(), MsgType::Reply_Get);
}

BOOST_AUTO_TEST_SUITE_END()

}  // namespace test
}  // namespace multiverso