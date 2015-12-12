#ifndef _MULTIVERSO_LOCKSYS_H_
#define _MULTIVERSO_LOCKSYS_H_

#include <vector>
#include <mutex>

// #if defined(_WIN32) || defined(_WIN64)
// #include <Windows.h>
// #endif

#include "conf.h"
using namespace std;

namespace multiverso
{
	class LockSys
	{
	public:
		// Initializes the lock system, creates the locks.
		static void Init();
        // Deletes the locks.
		static void Close();

		// Locks a table
		inline static void LockTable(int id)
		{
			// EnterCriticalSection(table_locks_[id % table_lock_num_]);
            table_locks_[id % table_lock_num_]->lock();
		}

		// Unlocks a table
		inline static void UnlockTable(int id)
		{
			//LeaveCriticalSection(table_locks_[id % table_lock_num_]);
            table_locks_[id % table_lock_num_]->unlock();
		}

		// Returns the table lock with id
		inline static mutex *GetTableLock(int id)
		{
			return table_locks_[id % table_lock_num_];
		}

		// Locks a row
		inline static void Lock(int id, bool lock_option = default_lock_option_)
		{
			if (lock_option)
			{
				// EnterCriticalSection(row_locks_[id % row_lock_num_]);
                row_locks_[id % row_lock_num_]->lock();
			}
		}

		// Unlocks a row
		inline static void Unlock(int id, bool lock_option = default_lock_option_)
		{
			if (lock_option)
			{
				// LeaveCriticalSection(row_locks_[id % row_lock_num_]);
                row_locks_[id % row_lock_num_]->unlock();
			}
		}

		// Returns the row lock with id
		inline static mutex *GetLock(int id)
		{
			return row_locks_[id % row_lock_num_];
		}

	private:
		// Creates the locks and adds into the container
		static void CreateLocks(int lock_num, vector<mutex*> &locks);
		// Closes all locks
		static void Close(vector<mutex*> &locks);

		// table lock config
		static int table_lock_num_;
		// static vector<CRITICAL_SECTION*> table_locks_;
        static vector<mutex*> table_locks_;
		// row lock config
		static int row_lock_num_;
		// static vector<CRITICAL_SECTION*> row_locks_;
		static vector<mutex*> row_locks_;
		static bool default_lock_option_;
		// assitant
		static mutex init_lock_;
		static bool is_initialized_;
	};

	class CSLock
	{
	public:
		CSLock(mutex *lock, bool lock_option = true)
		{
			lock_ = lock;
			lock_option_ = lock_option;
			if (lock_option_)
			{
				//EnterCriticalSection(lock_);
                lock_->lock();
			}
		}

		~CSLock()
		{
			if (lock_option_)
			{
				// LeaveCriticalSection(lock_);
                lock_->unlock();
			}
		}

	private:
		mutex *lock_;
		bool lock_option_;
	};
}

#endif // _MULTIVERSO_LOCKSYS_H_