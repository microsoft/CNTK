#ifndef _MULTIVERSO_CACHE_H_
#define _MULTIVERSO_CACHE_H_

#include <vector>
#include <unordered_map>
#include <mutex>
#include "table.h"
using namespace std;

namespace multiverso
{
    // The Cache class maintains a list of tables.
	class Cache
	{
	public:
		~Cache();

        // Returns a pointer to the specified table.
        TableBase *GetTable(int table_id);

		// The Adaptor DOES NOT create its cache by constructor directly as
		// serveral adaptors may share the same cache with identical cache ID.
		// The Adaptor gets its cache from this static method.
		static Cache *CreateCache(int cache_id);
        static void Clear();

        // The static cache_pool_ maintains the cache id -> cache mapping.
		static unordered_map<int, Cache*> cache_pool_;

	private:
		Cache();

		vector<TableBase*> tables_;

        // Initialization lock for avoiding different theads creating caches
        // with identical id.
		static mutex init_lock_;
	};
}

#endif // _MULTIVERSO_CACHE_H_