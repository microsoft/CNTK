#ifndef _MULTIVERSO_CONFIG_H_
#define _MULTIVERSO_CONFIG_H_

#include <iostream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <random>
using namespace std;

namespace YAML
{
    class Node;
}

namespace multiverso
{
    // the maximal data block size to be sent by MPI
    const long long BLOCK_SIZE = 1 << 27; // 128 MB
    const long long ADAPTOR_ADD_CACHE_SIZE = 1 << 25;  // 32 MB
    const long long MPI_SEND_BAR = 1 << 10;    // 1024

#pragma region configuration options
    // The communication type. REDUCE by default.
	enum class CommType : int
	{
		REDUCE = 0,
		P2P = 1,
		UNKNOWN = -1
	};

	// The local cache storage type
	enum class CacheType : int
	{
		STATIC = 0,
		DYNAMIC = 1,
		UNKNOWN = -1
	};

	// The synchronization type. We only support SYNC and ASYNC at this time,
	// should support stale sync in the future.
	enum class SyncType : int
	{
		SYNC = 0,
		ASYNC = 1,
        SSYNC = 2,
        AVERAGE = 3,
		UNKNOWN = -1
	};

	// Table element types.
	enum class EleType : int
	{
		INT = 0,
		FLOAT = 1,
		DOUBLE = 2,
        LONGLONG = 3,
		UNKNOWN = -1
	};

	// Table format: dense table or sparse table.
	enum class TableFormat : int
	{
		DENSE = 0,
		SPARSE = 1,
		UNKNOWN = -1
	};

    // ATTENTION(@developer): may deprecated this option.
	// Update type. 
	// DELTA: adding the delta to the global parameter server directly;
	// AVG: adding delta/client_count to the global parameter server.
	enum class UpdateType : int
	{
		DELTA = 0,      // adding delta directly
		DELTA_K = 1,	// adding 1/k * delta
		REPLACE = 2,	// replacing the values
		UNKNOWN = -1
	};

	// Remote request types
	enum class RequestType : int
	{
		REGISTER = 1,       // register
		REPLYREG = -1,      // replied register
		GET = 2,            // get data
		REPLYGET = -2,      // replied get
		ADD = 3,            // add/increase
		CLOCK = 4,          // submit an iteration
        REPLYCLOCK= -4,     // replied clock
		REDUCEANDBCAST = 5, // reduce and broadcast
		EXIT = 6,           // exit
		BARRIER = 7,        // barrier
		REPLYBARRIER = -7,  // replied barrier
		DATA = 8,			// get block index
		REPLYDATA = -8,			// get block index
		FINISH = 9,			// finish train
        UNKNOWN = 0
	};

    // An assistant class providing static methods of parsing configuration
    // strings into build-in configuration options.
    class OptionParser
    {
    public:
        static CommType GetCommType(string comm_type);
        static CacheType GetCacheType(string cache_type);
        static SyncType GetSyncType(string sync_type);
        static EleType GetEleType(string type_name);
        static TableFormat GetTableFormat(string format);
        static UpdateType GetUpdateType(string update_type);

    private:
        // Regularize the string with upper letters
        static void Upper(string *str);
    };
#pragma endregion // configuration options

#pragma region configuration data structures
	// Data structure storing training algorithm config.
	struct AlgoConfig
	{
        AlgoConfig() {}
		AlgoConfig(const YAML::Node &configroot);

		CommType comm_type;			// communication type
		CacheType cache_type;		// local cache type
		SyncType sync_type;         // synchronous type
		int delay_bound;            // worker progress bound
		int staleness;              // local parameter staleness
		// int expected_client_count;  // expected client number when starting

	private:
		// Throws exception if there is anything unexpected.
        void Check();
	};

	// Data structure storing multiverso server config
	struct ServerConfig
	{
        ServerConfig() {}
		ServerConfig(const YAML::Node &configroot, int process_count);

        vector<int> lookup_server2rank;
	};

	// Data structure storing multiverso lock config
	struct LockConfig
	{
        LockConfig() {}
		LockConfig(const YAML::Node &configroot);

        bool lock_option;
        int row_lock_count;
        int table_lock_count;

	private:
		// Throws exception if invalid variable
        void Check();
	};

	// Data structure storing table information.
	struct TableConfig
	{
        TableConfig() : table_id(-1) {}
        TableConfig(const YAML::Node &table_config);
        //TableConfig(YAML::Node *table_config);

		int table_id;
		long long row_count;
		long long col_count;
		EleType ele_type;
		TableFormat format;
		long long cache_size;
        // ATTENTION(@developers): may deprecate these 3 members
        UpdateType update_type;
        double init_lower;
        double init_upper;

	private:
		// Throws exception if there is anything invalid.
        void Check();
	};

    struct CheckpointInfo
    {
        CheckpointInfo();
        CheckpointInfo(const YAML::Node &checkpoint);

        int server_id;
        int minutes;
        string store_path;
        string recover_file;
    };
#pragma endregion // configuration data structures

    class PreConfig
    {
    public:
        PreConfig();
        ~PreConfig();

        YAML::Node *GetConfigNode() { return config_node_; }

        void SetServer(int *server_ranks, int server_count);
        void SetCommType(string comm_type);
        void SetSyncType(string sync_type);
        void SetDelayBound(int delay);
        void SetStaleness(int staleness);
        void SetLock(bool lock_option, int row_locks, int table_locks);
        void SetTable(int table_id, long long rows, long long cols, 
            string ele_type = "float");
        void SetCheckpoint(int server_id, int minutes = -1, 
            string store_path = "", string recover_file = "");

    private:
        YAML::Node *config_node_;
    };

    class Config
    {
    public:
        Config(string config_file, int process_count);
        Config(PreConfig *preconfig, int process_count);

        CommType GetCommType() { return algo_config_.comm_type; }
        SyncType GetSyncType() { return algo_config_.sync_type; }
        CacheType GetCacheType() { return algo_config_.cache_type; }
        int GetDelay() { return algo_config_.delay_bound; }
        int GetServerCount();
        int GetServerId(int rank);
        int GetServerRank(int server_id);
        int GetTableCount();
        TableConfig GetTableConfig(int table_id);
        LockConfig GetLockConfig();
        CheckpointInfo GetCheckpointInfo(int server_id);

    private:
        void Parse(const YAML::Node &config_root, int process_count);
        void ParseTableConfig(const YAML::Node &configroot);
        void ParseCheckpointConfig(const YAML::Node &configroot, int size);
        long long ConfigHash();

        // configuration properties
        AlgoConfig algo_config_;
        ServerConfig server_config_;
        LockConfig lock_config_;
        vector<TableConfig> table_configs_;
        vector<CheckpointInfo> checkpoints_;
    };
}

#endif // _MULTIVERSO_CONFIG_H_