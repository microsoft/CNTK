#include <multiverso/multiverso.h>
#include <multiverso/net.h>
#include <multiverso/util/log.h>
#include <multiverso/util/net_util.h>

namespace multiverso {
namespace test {

void TestNet(int argc, char* argv[]) {
  NetInterface* net = NetInterface::Get();
  net->Init(&argc, argv);

  const char* chi1 = std::string("hello, world").c_str();
  const char* chi2 = std::string("hello, c++").c_str();
  const char* chi3 = std::string("hello, multiverso").c_str();
  char* hi1 = new char[14];

#ifdef _MSC_VER
  strcpy_s(hi1, 14, chi1);
#else
  strcpy(hi1, chi1);
#endif

  char* hi2 = new char[12];
#ifdef _MSC_VER
  strcpy_s(hi2, 12, chi2);
#else
  strcpy(hi2, chi2);
#endif

  char* hi3 = new char[19];
#ifdef _MSC_VER
  strcpy_s(hi3, 19, chi3);
#else
  strcpy(hi3, chi3);
#endif

  if (net->rank() == 0) {
    for (int rank = 1; rank < net->size(); ++rank) {
      MessagePtr msg(new Message());
      msg->set_src(0);
      msg->set_dst(rank);
      msg->Push(Blob(hi1, 13));
      msg->Push(Blob(hi2, 11));
      msg->Push(Blob(hi3, 18));
      for (int i = 0; i < msg->size(); ++i) {
        Log::Info("In Send: %s\n", msg->data()[i].data());
      };
      while (net->Send(msg) == 0);
      Log::Info("rank 0 send\n");
    }

    for (int i = 1; i < net->size(); ++i) {
      MessagePtr msg(new Message());
      msg.reset(new Message());
      while (net->Recv(&msg) == 0) {
        // Log::Info("recv return 0\n");
      }
      Log::Info("rank 0 recv\n");

      std::vector<Blob> recv_data = msg->data();
      CHECK(recv_data.size() == 3);
      for (int i = 0; i < msg->size(); ++i) {
        Log::Info("recv from srv %d: %s\n", msg->src(), recv_data[i].data());
      };
    }
  }
  else {// other rank
    MessagePtr msg(new Message());
    while (net->Recv(&msg) == 0) {
      // Log::Info("recv return 0\n");
    }
    Log::Info("rank %d recv\n", net->rank());
    std::vector<Blob>& recv_data = msg->data();
    CHECK(recv_data.size() == 3);
    for (int i = 0; i < msg->size(); ++i) {
      Log::Info("%s\n", recv_data[i].data());
    }

    msg.reset(new Message());
    msg->set_src(net->rank());
    msg->set_dst(0);
    msg->Push(Blob(hi1, 13));
    msg->Push(Blob(hi2, 11));
    msg->Push(Blob(hi3, 18));
    while (net->Send(msg) == 0);
    Log::Info("rank %d send\n", net->rank());
  }
  net->Finalize();
}

}  // namespace test
}  // namespace multiverso