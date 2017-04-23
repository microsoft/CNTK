#pragma once

#include <multiverso/table/matrix_table.h>
#include <vector>

using namespace System;

namespace MultiversoCLR {

  interface class IWorkerTable {
  public:
    static IWorkerTable^ CreateTable(int table_id, int num_rows, int num_cols, System::String^ type);
    void Get(int row_id, void* buffer, int size);
    void Get(void* buffer, int size);
    void Get(array<int>^ row_ids, array<void*>^ buffers, int size);

    void Add(int row_id, void* buffer, int size);
    void Add(void* buffer, int size);
    void Add(array<int>^ row_ids, array<void*>^ buffers, int size);
  };

  template <class Type>
  public ref class MatrixTable : public IWorkerTable {
  public:
    MatrixTable(int num_rows, int num_cols) {
      multiverso::MatrixTableOption<Type> option(num_rows, num_cols);
      table_ = multiverso::MV_CreateTable(option);
    }

    ~MatrixTable() {
      delete table_;
    }

    virtual void Get(void* buffer, int size) {
      table_->Get(static_cast<Type*>(buffer), size);
    }

    virtual void Get(int row_id, void* buffer, int size) {
      table_->Get(row_id, static_cast<Type*>(buffer), size);
    }

    virtual void Get(array<int>^ row_ids, array<void*>^ buffers, int size) {
      std::vector<int>  row_id_vec(size);
      std::vector<Type*> buffer_vec;
      pin_ptr<int> p = &row_ids[0];
      memcpy(row_id_vec.data(), p, size * sizeof(int));
      for (int i = 0; i < size; ++i) {
        buffer_vec.push_back(static_cast<Type*>(buffers[i]));
      }
      table_->Get(row_id_vec, buffer_vec, size);
    }

    virtual void Add(int row_id, void* buffer, int size) {
      table_->Add(row_id, static_cast<Type*>(buffer), size);
    }

    virtual void Add(void* buffer, int size) {
      table_->Add(static_cast<Type*>(buffer), size);
    }

    virtual void Add(array<int>^ row_ids, array<void*>^ buffers, int size) {
      std::vector<int>  row_id_vec(size);
      std::vector<Type*> buffer_vec;
      pin_ptr<int> p = &row_ids[0];
      memcpy(row_id_vec.data(), p, size * sizeof(int));
      for (int i = 0; i < size; ++i) {
        buffer_vec.push_back(static_cast<Type*>(buffers[i]));
      }
      table_->Add(row_id_vec, buffer_vec, size);
    }
  private:
    multiverso::MatrixWorkerTable<Type>* table_;
  };

  IWorkerTable^ IWorkerTable::CreateTable(int, int num_rows, int num_cols, System::String^ type) {
    if (type->Equals("Int"))    return gcnew MatrixTable<int>(num_rows, num_cols);
    if (type->Equals("Float"))  return gcnew MatrixTable<float>(num_rows, num_cols);
    if (type->Equals("Double")) return gcnew MatrixTable<double>(num_rows, num_cols);
    throw gcnew Exception("Element Type " + type + " not implemented");
  }
}