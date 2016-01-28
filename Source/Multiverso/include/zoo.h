#ifndef _MULTIVERSO_ZOO_H_
#define _MULTIVERSO_ZOO_H_

#include <thread>
#include "conf.h"
#include "server_base.h"
#include "comm.h"
using namespace std;

namespace multiverso
{
    class Zoo
    {
    public:
		static void InitMPI(int *argc, char **argv[]);
		static void InitMPI();
		static void InitMutliversoSDK(int local_worker_count, string config_file);
		
		static void FinishTrain();
        static void Close(bool finalize_mpi);

        // dashboard components
        static PreConfig *preconfig_;
        static Config *config_;

        static ServerBase *server_;
        static thread *server_thread_;

        static Communicator *comm_;
        static thread *comm_thread_;
    };
}

#endif  // _MULTIVERSO_ZOO_H_