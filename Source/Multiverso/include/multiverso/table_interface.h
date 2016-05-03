#ifndef MULTIVERSO_TABLE_INTERFACE_H_
#define MULTIVERSO_TABLE_INTERFACE_H_

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cctype>

#include "multiverso/blob.h"
#include "multiverso/io/io.h"

namespace multiverso {

typedef int32_t integer_t;

class Waiter;
struct AddOption;
struct GetOption;

// User implementent this
class WorkerTable {
public:
  WorkerTable();
  virtual ~WorkerTable() = default;

  void Get(Blob keys, const GetOption* option = nullptr);
  void Add(Blob keys, Blob values, const AddOption* option = nullptr);

  int GetAsync(Blob keys, const GetOption* option = nullptr);
  int AddAsync(Blob keys, Blob values, const AddOption* option = nullptr);

  void Wait(int id);

  void Reset(int msg_id, int num_wait);

  void Notify(int id);

  virtual int Partition(const std::vector<Blob>& kv,
    std::unordered_map<int, std::vector<Blob> >* out) = 0;

  virtual void ProcessReplyGet(std::vector<Blob>&) = 0;

  // add user defined data structure
private:
  std::string table_name_;
  // assuming there are at most 2^32 tables
  int table_id_;
  std::mutex m_;
  std::vector<Waiter*> waitings_;
  int msg_id_;
};

// TODO(feiga): move to a seperate file
class Stream;

// interface for checkpoint table
class Serializable {
public:
  virtual void Store(Stream* s) = 0;
  virtual void Load(Stream* s) = 0;
};

// discribe the server parameter storage data structure and related method
class ServerTable : public Serializable {
public:
  ServerTable();
  virtual ~ServerTable() = default;
  virtual void ProcessAdd(const std::vector<Blob>& data) = 0;
  virtual void ProcessGet(const std::vector<Blob>& data,
                          std::vector<Blob>* result) = 0;
};

// TODO(feiga): provide better table creator method
// Abstract Factory to create server and worker
// my new implementation
class TableFactory {
public:
  template<typename Key, typename Val = void>
  static WorkerTable* CreateTable(const std::string& table_type,
    const std::vector<void*>& table_args,
    const std::string& dump_file_path = "");
  virtual ~TableFactory() {}
protected:
  virtual WorkerTable* CreateWorkerTable() = 0;
  virtual ServerTable* CreateServerTable() = 0;
};

// older one
class TableHelper {
public:
  TableHelper() {}
  WorkerTable* CreateTable();
  virtual ~TableHelper() {}

protected:
  virtual WorkerTable* CreateWorkerTable() = 0;
  virtual ServerTable* CreateServerTable() = 0;
};

// template<typename T>
// class MatrixTableFactory;
// template function should be defined in the same file with declaration
// template<typename Key, typename Val>
// WorkerTable* TableFactory::CreateTable(const std::string& table_type,
//  const std::vector<void*>& table_args, const std::string& dump_file_path) {
//  bool worker = (MV_WorkerId() >= 0);
//  bool server = (MV_ServerId() >= 0);
//  TableFactory* factory;
//  if (table_type == "matrix") {
//    factory = new MatrixTableFactory<Key>(table_args);
//  }
//  else if (table_type == "array") {
//  }
//  else CHECK(false);
//
//  if (server) factory->CreateServerTable();
//  if (worker) return factory->CreateWorkerTable();
//  return nullptr;
// }

}  // namespace multiverso

#endif  // MULTIVERSO_TABLE_INTERFACE_H_
