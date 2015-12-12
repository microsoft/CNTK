// adaptor.h: Declares the main user interface Adaptor class.

#ifndef _MULTIVERSO_ADAPTOR_H_
#define _MULTIVERSO_ADAPTOR_H_

#include <vector>
#include <unordered_map>
#include <thread>
using namespace std;

namespace zmq
{
    class socket_t;
}

namespace multiverso
{
	class Cache;

	class Adaptor
	{
	public:
        // Creates and Adaptor and its cache. It will be regarded as working
        // adaptor if adaptor_id >= 0, or monitor otherwise.
		Adaptor(int adaptor_id, int cache_id);
        ~Adaptor();

        // Loads the specified rows into local cache in a batch mode.
        // Will load the whole table if rows is empty.
        // BatchLoad methods only work for P2P comm mode and thery are 
        // transparent to REDUCE comm mode.
        long long BatchLoad(int table_id, vector<long long> &rows);
        long long BatchLoad(int table_id, long long *rows = nullptr, 
            long long num = 0);
        // Returns the memory of the row.
		void *Get(int table_id, long long row_id);
        // Updates a row by adding a delta.
        void Add(int table_id, long long row_id, void *delta, 
            float server_coef = 1.0f);
        // Updates a row by replacing the value.
        void Set(int table_id, long long row_id, void *value);

        // All working adaptor sync at this point and continue.
        void Barrier();

        void Clock();

    private:
		// Registers the client adaptor to communicator.
		void Register();
        void WaitForReply(int wait_count, zmq::socket_t *socket);
        // Send all and clean up the local add cache.
        void SendAddCache();
        int GetRowServerId(long long row_id) { return row_id % server_count_; }

        // Sync the model with AllReduce method.
        void AllReduce();

        // properties
		int adaptor_id_;
		int cache_id_;
        int server_count_;
        int table_count_;
		Cache *cache_;

        // A simple memory pool method of caching and aggregating the updates
        // to servers.
        vector<unordered_map<long long, char*>*> add_records_;
        char *add_cache_;
        long long used_add_size_;

		thread::id * _tid;
	};
}

#endif // _MULTIVERSO_ADAPTOR_H_