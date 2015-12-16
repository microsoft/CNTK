// adaptor.h: Declares the main user interface Adaptor class.

#ifndef _MULTIVERSO_ADAPTOR_H_
#define _MULTIVERSO_ADAPTOR_H_

#include <vector>
#include <unordered_map>
#include <thread>
#include <cstdint>
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
		int64_t BatchLoad(int table_id, vector<int64_t> &rows);
		int64_t BatchLoad(int table_id, int64_t *rows = nullptr,
			int64_t num = 0);
		int64_t BatchLoad(int table_id, void *dst, uint64_t *idx_each_server, uint64_t *size_each_server);
		// Returns the memory of the row.
		void *Get(int table_id, int64_t row_id);
		// Updates a row by adding a delta.
		void Add(int table_id, int64_t row_id, void *delta,
			float server_coef = 1.0f);
		// Updates a row by replacing the value.
		void Set(int table_id, int64_t row_id, void *value);

		// All working adaptor sync at this point and continue.
		void Barrier();

		void Clock();

		void Update(void *dst, uint64_t *idx_each_server, uint64_t *size_each_server, int table_id, int64_t rows, void *delta);

	private:
		// Registers the client adaptor to communicator.
		void Register();
		void WaitForReply(int wait_count, zmq::socket_t *socket);
		// Send all and clean up the local add cache.
		void SendAddCache();
		int GetRowServerId(int64_t row_id) { return row_id % server_count_; }

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
		vector<unordered_map<int64_t, char*>*> add_records_;
		char *add_cache_;
		int64_t used_add_size_;

		thread::id * _tid;
	};
}

#endif // _MULTIVERSO_ADAPTOR_H_