#ifndef _MULTIVERSO_MULTIVERSO_H_
#define _MULTIVERSO_MULTIVERSO_H_

#include <string>
#include "server_base.h"
#include "adaptor.h"
using namespace std;

namespace multiverso
{
    //Sets the servers. server_ranks is an array specifying the corresponding
    // mpi ranks of each server.
    void SetServer(int *server_ranks, int server_count);

    // Sets the communication types for passing the updates, options: 
    // "reduce": uses MPI Allreduce;
    // "p2p": uses MPI non-blocking p2p communication.
    void SetCommType(string comm_type = "reduce");

    // Sets synchronization types. Options:
    // "sync": 
    // "async":
    // "ssync":
    // "average":
    void SetSyncType(string sync_type = "sync");

    // Sets the client adaptor progress delay bound. Multiverso ensure that the
    // progress difference between the fastest and slowest adaptor should not
    // greater than this delay bound.
    void SetDelayBound(int delay = 0);

    // @TODO: add comment
    void SetStaleness(int staleness = 0);

    // Adds a table with specifying its properties:
    // table_id: an integer identifying the table;
    // rows: number of rows
    // cols: number of columns
    // ele_type: element data type. "float" by default, could be "double"
    void SetTable(int table_id, long long rows, long long cols, 
        string ele_type = "float");

    // Sets the local cache locks. No lock by defualt as each adaptor uses its
    // own cache. You may consider setting the locks if multiple caches will
    // share a same cache.
    void SetLock(bool lock_option = false, int row_locks = 100, 
        int table_locks = 1);

    void SetCheckpoint(int server_id, int minutes = -1, string store_path = "", 
        string recover_file = "");

    // A user can reimplement the ServerBase class for customized server logic.
    // A default server instances will be executed if no setting this option.
    void SetCustomizedServer(ServerBase *customized_server);

    // Globally initialization. Initializing MPI environment; creating
    // background communication thread; creating a background server thread if
    // necessary.
    void Init(int local_worker_count);

	void InitMPI(int *argc, char** argv[]);

	void InitMPI();

    // Closes the Multiverso environments. If finalized_mpi is true, the MPI 
    // environment will be closed as well, or the MPI environment will be kept
    // and can be used in another round.
	void FinishTrain();

    void Close(bool finalize_mpi = true);

	// Returns the total number of processes in current MPI world.
	int GetMPISize();
	// Returns the MPI rank of the current process.
	int GetMPIRank();
	
	void SetLog(bool is_enable);
	bool IsModelOwner();
}

#endif // _MULTIVERSO_MULTIVERSO_H_