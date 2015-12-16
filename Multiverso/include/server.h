#ifndef _MULTIVERSO_SERVER_H_
#define _MULTIVERSO_SERVER_H_

#include "table.h"
#include "zmq_util.h"
#include "server_base.h"
#include "mt_queue.h"

namespace multiverso
{
    class Server : public ServerBase
    {
    public:
        Server(int server_id, const CheckpointInfo &cp_info);

        void Start() override;
        void Stop() override;
        bool IsWorking() override { return is_working_; }

    private:
        void Init();
        void Clear();
        void Checkpoint();
        void LoadCheckpoint();

        void Process_Get(shared_ptr<MsgPack> request);
        void Process_Add(shared_ptr<MsgPack> request);

        bool is_working_;
        int server_id_;
        int server_count_;
        vector<TableBase*> tables_;

        zmq::socket_t *router_;
        zmq::pollitem_t *poll_items_;
        int poll_count_;

        CheckpointInfo checkpoint_info_;
        thread *checkpoint_thread_;
		thread *add_thread_;
		MtQueueMove<shared_ptr<MsgPack>> add_queue_;
        bool is_checkpoint_;
        bool is_checkpoint_working_;

		std::vector<std::pair<int, int>> receive_log_;
    };
}

#endif // _MULTIVERSO_SERVER_H_ 