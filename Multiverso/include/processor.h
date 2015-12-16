#ifndef _MULTIVERSO_PROCESSOR_H_
#define _MULTIVERSO_PROCESSOR_H_

#include <memory>
#include "zmq_util.h"
#include "msg_pack.h"
#include "cache.h"
using namespace std;

namespace multiverso
{
    class ProcessorBase
    {
    public:
        ProcessorBase(zmq::socket_t *socket4worker, zmq::socket_t *socket4server);

    protected:
        zmq::socket_t *socket4worker_;
        zmq::socket_t *socket4server_;
        vector<shared_ptr<MsgPack>> tasks_;
    };

    class Register : public ProcessorBase
    {
    public:
        Register(zmq::socket_t *socket4worker, int exp_local_worker_count);
        void Process(shared_ptr<MsgPack> request);
        void ProcessRepliedRegister(shared_ptr<MsgPack> reply);
        int GetLocalAdaptorCount() { return local_adaptor_count_; }
        int GetGlobalAdaptorCount() { return global_adaptor_count_; }

    protected:
        void ReplyOuterRegister(shared_ptr<MsgPack> request);
        void ReplyLocalRegister(shared_ptr<MsgPack> request);

        int exp_local_adaptor_count_;
        int local_adaptor_count_;
        int exp_global_adaptor_count_;
        int global_adaptor_count_;
    };

    class AllreduceProcessor : public ProcessorBase
    {
    public:
        AllreduceProcessor(zmq::socket_t *socket4worker, 
            Register *register_processor);
        void Process(shared_ptr<MsgPack> request, vector<TableBase*> &tables);
        void FakeProcess(vector<TableBase*> &tables, int &global_finished);

    protected:
        void Allreduce(vector<TableBase*> &tables, int &global_finished);
        void ReplyAll();

        int valid_worker_count_;
        Register *register_processor_;

        vector<int> cache_ids_;
    };

    class AddProcessor : public ProcessorBase
    {
    public:
        AddProcessor(zmq::socket_t *socket4server)
            : ProcessorBase(nullptr, socket4server) {}
        void Process(shared_ptr<MsgPack> request, vector<TableBase*> &tables);

    private:
        void AddToCommTable(shared_ptr<MsgPack> request, 
            vector<TableBase*> &tables);
    };

    class ExitProcessor : public ProcessorBase
    {
    public: 
        ExitProcessor(zmq::socket_t *socket4worker);
        void Process(shared_ptr<MsgPack> request);

    private:
        int exit_count_;
        shared_ptr<MsgPack> local_exit_msg_;
    };

    class ClockProcessor : public ProcessorBase
    {
    public:
        ClockProcessor(zmq::socket_t *socket4worker);
        void Process(shared_ptr<MsgPack> request, int global_adaptor_count);
        void ProcessReply(shared_ptr<MsgPack> reply);
    private:
        vector<int> clocks_;
    };

    class BarrierProcessor : public ProcessorBase
    {
    public:
        BarrierProcessor(zmq::socket_t *socket4worker);
        void Process(shared_ptr<MsgPack> request, int global_adaptor_count);
        void ProcessReply(shared_ptr<MsgPack> reply);
    private:
        int recv_count_;
    };
}

#endif // _MULTIVERSO_PROCESSOR_H_ 