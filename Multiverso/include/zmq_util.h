#ifndef _MULTIVERSO_ZMQ_UTIL_H_
#define _MULTIVERSO_ZMQ_UTIL_H_

#include <queue>
#include <unordered_map>
#include <thread>
#include <memory>
#include "msg_pack.h"
using namespace std;

#define COMM_ENDPOINT "inproc://comm"
#define SERVER_ENDPOINT "inproc://server"

namespace multiverso
{
    const int ZMQ_IO_THREAD_NUM = 1;
    const int ZMQ_POLL_TIMEOUT = 0;

    class ZMQUtil
    {
    public:
        static zmq::context_t *GetZMQContext();
        // Returns a zmq DEALER socket bound to current thread.
        static zmq::socket_t *GetThreadSocket(thread::id *tid = nullptr);
        static void Clear();

        static void ZMQPoll(
            zmq::pollitem_t *poll_items,
            int poll_count,
            const vector<zmq::socket_t*> &sockets,
            queue<shared_ptr<MsgPack>> &msgs_queue);

    private:
        // the global zmq context
        static zmq::context_t *zmq_context_;
        // maitaining the zmq DEALER sockets for each thread
        static unordered_map<thread::id, zmq::socket_t*> thread_sockets_;
    };
}

#endif // _MULTIVERSO_ZMQ_UTIL_H_ 