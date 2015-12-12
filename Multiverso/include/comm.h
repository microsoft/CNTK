#ifndef _MULTIVERSO_COMM_H_
#define _MULTIVERSO_COMM_H_

#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include "cache.h"
#include "msg_pack.h"
#include "processor.h"
using namespace std;

namespace multiverso
{
#define STOP_TIME 1000

    // The Communicator class is responsible for monitoring the request and
    // reply messages from inner threads or outer processes and transfering
    // them to appropriate end receivers.
    class Communicator
    {
    public:
        Communicator();

        // A background thead starts running the communication routines.
        void Start(int exp_local_worker_count);
        // Returns if the Communicator starts working.
        bool IsWorking() { return is_working_; }
        // Stops the running communication threads.
        void Stop();

		bool IsLast() { return is_last_updater_; }
        //// TEST method: get the numbers of adaptors in different scopes.
        //static void Test_GetAdaptorCounts(int *exp_local_count,
        //    int *local_count, int *global_count)
        //{
        //    *exp_local_count = 0; //exp_local_adaptor_count_;
        //    *local_count = register_processor_->GetLocalAdaptorCount();
        //    *global_count = register_processor_->GetGlobalAdaptorCount();
        //}

    private:
        // Initializes the communication variables.
        void Init(int exp_local_worker_count);
        void Clear();

        // Main function of processing the messages.
        void ProcessRequest(shared_ptr<MsgPack> request);
        void SendToServer(shared_ptr<MsgPack> request);
		void SendToWorker(shared_ptr<MsgPack> reply);

        void ProcessExit(shared_ptr<MsgPack> request);

        //static void Process_Clock(MsgPack *request);

        //-- BEGIN: properties -----------------------------------------------/
        bool is_working_;
		bool is_last_updater_;
        vector<TableBase*> tables_;

        // the communication thread ZMQ_ROUTER socket
        zmq::socket_t *router_;  // comm with working threads
        zmq::socket_t *dealer_;  // comm with server
        zmq::pollitem_t *pollitems_;
        int pollitem_count_;

        Register *register_processor_;
        AllreduceProcessor *allreduce_processor_;
        AddProcessor *add_processor_;
        ExitProcessor *exit_processor_;
        ClockProcessor *clock_processor_;
        BarrierProcessor *barrier_processor_;
        //-- END: properties -------------------------------------------------/
    };
}

#endif // _MULTIVERSO_COMM_H_