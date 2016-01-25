#ifndef _MULTIVERSO_MPI_UTIL_H_
#define _MULTIVERSO_MPI_UTIL_H_

#include <queue>
#include <memory>
#include "mpi.h"
#include "conf.h"
#include "msg_pack.h"
using namespace std;

namespace multiverso
{
    enum class MPIOperator 
    {
        SUM = 0,
        BAND = 1,
        BOR = 2
    };

    class MPIUtil
    {
    public:
        static void Init(int *argc, char **argv[]);
		static void Init();
        static void Close();

        static int GetMPIRank() { return mpi_rank_; }
        static int GetMPISize() { return mpi_size_; }
		static char* GetNodeName() { return node_name_; }
        static shared_ptr<MsgPack> MPIProbe();
        static void Send(shared_ptr<MsgPack> msg_pack, int dst_rank);
        static int64_t SendQueueSize() { return send_msg_queue_.size(); }
        static void Allreduce(void *sendbuf, void *recvbuf, int count, 
            EleType ele_type, MPIOperator op = MPIOperator::SUM);

    private:
        static MPI_Datatype GetMPIType(EleType ele_type);
        static void AdjustBufferSize(char **buffer, int *buffer_size, int size);

        static int mpi_rank_;
        static int mpi_size_;

        static int send_buffer_size_;
        static char *send_buffer_;
		static char *node_name_;
        static MPI_Request send_request_;
        static queue<shared_ptr<MsgPack>> send_msg_queue_;
        static queue<int> send_dst_rank_queue_;
        

        static int recv_buffer_size_;
        static char *recv_buffer_;
        static MPI_Request recv_request_;
    };
}

#endif // _MULTIVERSO_MPI_UTIL_H_