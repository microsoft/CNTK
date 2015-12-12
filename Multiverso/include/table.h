// Tracking info:
// feiyan, 2014-09-19, creating the sketch
// v-guihom, 2014-09-22, editing the parameter for construct function
// v-guihom, 2014-09-24, doing some correction

#ifndef _MULTIVERSO_TABLE_H_
#define _MULTIVERSO_TABLE_H_

#include <random>
#include "conf.h"
using namespace std;

namespace multiverso
{
	// Abstract base Table class. Different instantiable Table class derives
	// from this interface.
	class TableBase
	{
	public:
		virtual ~TableBase() {}
        // Returns a pointer to the specified row.
		virtual void *Get(long long row_id) = 0;
        virtual void *GetAll() = 0;
        // Adds a delta to the row.
        virtual int Add(long long row_id, void* delta) = 0;
        // Fills the specified row.
		virtual int Set(long long row_id, void *row) = 0;
        // Fills the whole table.
		virtual int FullFill(void *src) = 0;
        virtual int SetZero() = 0;
        virtual int Scale(float coef) = 0;

        virtual long long GetElementSize() = 0;
        virtual long long GetMemSize() = 0;
        virtual long long GetRowMemSize() = 0;
        virtual long long GetRowCount() { return table_config_.row_count; }
        virtual long long GetColCount() { return table_config_.col_count; }

        virtual void OuterScale(void *result, float coef = 1) = 0;
        virtual void OuterAdd(void *result, void *operand, float coef = 1) = 0;

    protected:
		// Common data members
		TableConfig table_config_;

		// A shared static random engine for value initialization.
		static default_random_engine rand_eng_;
	};


    // The StaticTable class maintains a continuous block of memory as a table.
	template <typename T>
	class StaticTable : public TableBase
	{
	public:
        StaticTable(int table_id, int mod = 1, int rm = 0);
		~StaticTable();

        void *Get(long long row_id) override;
        inline void *GetAll() override { return data_; }
        int Add(long long row_id, void *delta) override;
        int Set(long long row_id, void *value) override;
        int FullFill(void *value) override;
        int SetZero() override;
        int Scale(float coef) override;

        long long GetElementSize() override { return sizeof(T); }
        long long GetMemSize() override { return phy_mem_size_; }
        long long GetRowMemSize() override { return row_mem_size_; }

        void OuterScale(void *result, float coef = 1.0f) override;
        void OuterAdd(void *result, void *operand, float coef = 1.0f) override;

	private:
        inline long long GetPhysicalRow(long long row_id) const { return row_id / mod_; }

		T *data_;   // a continuous memory blocks of storying the static table
        int mod_;
        int rm_;
        long long phy_row_count_;
        long long phy_size_;
        long long phy_mem_size_;
        long long row_mem_size_;
	};


    // The TableCreator class is a simple static factory class of creating and 
    // returning a table according to the config.
	class TableCreator
	{
	public:
        // Creates and returns a table.
        static TableBase *CreateTable(
            int table_id, 
            int mod = 1, 
            int rm = 0,
            bool is_server = false);
	};
}

#endif // _MULTIVERSO_TABLE_H_ 