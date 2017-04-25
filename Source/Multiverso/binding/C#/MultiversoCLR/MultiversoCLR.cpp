// This is the main DLL file.

#include "MultiversoCLR.h"
#include <multiverso/util/configure.h>

namespace {

  array<char>^ ToCString(String^ csString)
  {
    array<char>^ cString = gcnew array<char>(csString->Length + 1);
    int indx = 0;
    for (; indx < csString->Length; indx++)
    {
      cString[indx] = (char)csString[indx];
    }
    cString[indx] = 0;
    return cString;
  }
}

namespace MultiversoCLR {

  bool MultiversoWrapper::NetBind(int rank, System::String^ endpoint)
  {
    array<char>^ ip_port = ToCString(endpoint);
    pin_ptr<char> ptr = &ip_port[0];
    return multiverso::MV_NetBind(rank, ptr) == 0;
  }

  bool MultiversoWrapper::NetConnect(array<int>^ ranks, array<System::String^>^ endpoints)
  {
    if (ranks->Length != endpoints->Length) {
      // Fatal error
    }
    pin_ptr<int> p_ranks = &ranks[0];
    array<char*>^ array_endpoints = gcnew array<char*>(endpoints->Length);
    for (int i = 0; i < endpoints->Length; ++i) {
      array<char>^ ip_port = ToCString(endpoints[i]);
      pin_ptr<char> ptr = &ip_port[0];
      array_endpoints[i] = ptr;
    }
    pin_ptr<char*> p_endpoints = &array_endpoints[0];
    return multiverso::MV_NetConnect(p_ranks, p_endpoints, ranks->Length) == 0;
  }

  void MultiversoWrapper::NetFinalize()
  {
    multiverso::MV_NetFinalize();
  }

  void MultiversoWrapper::Init(int num_tables, bool sync) {
    worker_tables_ = gcnew array<IWorkerTable^>(num_tables);
    if (sync) multiverso::SetCMDFlag("sync", true);
    Init();
  }
  void MultiversoWrapper::Shutdown() {
    // The false means finalize_net = false
    // We finalize the net resource separately by calling MultiversoWrapper.NetFinalize
    multiverso::MV_ShutDown(false);
  }

  int MultiversoWrapper::Rank() { return multiverso::MV_Rank(); }

  int MultiversoWrapper::Size() { return multiverso::MV_Size(); }

  void MultiversoWrapper::Barrier() { multiverso::MV_Barrier(); }

  void MultiversoWrapper::CreateTables(array<int>^ rows, array<int>^ cols, array<System::String^>^ eleTypes)
  {
    for (int i = 0; i < rows->Length; ++i)
    {
      CreateTable(i, rows[i], cols[i], eleTypes[i]);
    }
  }

  void MultiversoWrapper::CreateTable(int table_id, int rows, int cols, System::String^ eleType)
  {
    CreateWorkerTable(table_id, rows, cols, eleType);
  }

  generic <class Type>
    void MultiversoWrapper::Get(int table_id, array<Type>^ p_value)
    {
      pin_ptr<Type> p = &p_value[0];
      worker_tables_[table_id]->Get(p, p_value->Length);
    }

    generic <class Type>
      void MultiversoWrapper::Get(int table_id, int row_id, array<Type>^ p_value)
      {
        pin_ptr<Type> p = &p_value[0];
        worker_tables_[table_id]->Get(row_id, p, p_value->Length);
      }

      generic <class Type>
        void MultiversoWrapper::Add(int table_id, array<Type>^ p_update)
        {
          pin_ptr<Type> p = &p_update[0];
          worker_tables_[table_id]->Add(p, p_update->Length);
        }

        generic <class Type>
          void MultiversoWrapper::Add(int table_id, int row_id, array<Type>^ p_update)
          {
            pin_ptr<Type> p = &p_update[0];
            worker_tables_[table_id]->Add(row_id, p, p_update->Length);
          }

          void MultiversoWrapper::Init() { multiverso::MV_Init(nullptr, nullptr); }

          void MultiversoWrapper::CreateWorkerTable(int table_id, int rows, int cols, System::String^ eleType)
          {
            worker_tables_[table_id] = IWorkerTable::CreateTable(table_id, rows, cols, eleType);
          }
}
