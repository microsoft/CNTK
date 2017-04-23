// MultiversoCLR.h

#pragma once

// #include "ITable.h"
#include "MatrixTable.h"

using namespace System;

namespace MultiversoCLR {

  public ref class MultiversoWrapper
  {
  public:
    static bool NetBind(int rank, System::String^ endpoint);
    static bool NetConnect(array<int>^ ranks, array<System::String^>^ endpoints);
    static void NetFinalize();

    static void Init(int num_tables, bool sync);
    static void Shutdown();

    static void CreateTables(array<int>^ rows, array<int>^ cols, array<System::String^>^ eleTypes);
    static void CreateTable(int table_id, int rows, int cols, System::String^ eleType);

    static int Rank();
    static int Size();
    static void Barrier();

    generic <class Type>
      static void Get(int table_id, array<Type>^ p_value);

    generic <class Type>
    static void Get(int table_id, int row_id, array<Type>^ p_value);

    generic <class Type>
    static void Add(int table_id, array<Type>^ p_update);

    generic <class Type>
    static void Add(int table_id, int row_id, array<Type>^ p_value);

  private:
    static void Init();
    static void CreateWorkerTable(int table_id, int rows, int cols, System::String^ eleType);
    static array<IWorkerTable^>^ worker_tables_;
  };
}
