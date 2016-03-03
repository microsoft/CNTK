#ifndef MULTIVERSO_TABLE_INTERFACE_H_
#define MULTIVERSO_TABLE_INTERFACE_H_

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

#include "multiverso/blob.h"

namespace multiverso {

class Waiter;
// User implementent this
class WorkerTable {
public:
  WorkerTable();
  virtual ~WorkerTable() {}

  void Get(Blob keys) { Wait(GetAsync(keys)); }
  void Add(Blob keys, Blob values) { Wait(AddAsync(keys, values)); }

  // TODO(feiga): add call back
  int GetAsync(Blob keys); 
  int AddAsync(Blob keys, Blob values);

  void Wait(int id);

  void Reset(int msg_id, int num_wait);

  void Notify(int id);
  
  virtual int Partition(const std::vector<Blob>& kv,
    std::unordered_map<int, std::vector<Blob> >* out) = 0;

  virtual void ProcessReplyGet(std::vector<Blob>&) = 0;
  
  const std::string name() { return std::string(typeid(this).name());};

  // add user defined data structure
private:
  std::string table_name_;
  int table_id_;
  std::unordered_map<int, Waiter*> waitings_;
  std::atomic_int msg_id_;
};

// discribe the server parameter storage data structure and related method
class ServerTable {
public:
  ServerTable();
  virtual void ProcessAdd(const std::vector<Blob>& data) = 0;
  virtual void ProcessGet(const std::vector<Blob>& data,
    std::vector<Blob>* result) = 0;

  const std::string name() { return std::string(typeid(this).name());};
  
  // add user defined server process logic 
  void Process(const std::string instruction, const std::vector<Blob>& data, std::vector<Blob>* result = nullptr);

  // add user defined server storage data structure
};

// TODO(feiga): provide better table creator method
// Abstract Factory to create server and worker
class TableFactory {
  //  static TableFactory* GetTableFactory();
  virtual WorkerTable* CreateWorker() = 0;
  virtual ServerTable* CreateServer() = 0;
  static TableFactory* fatory_;
};

namespace table {

}

class TableBuilder {
public:
  TableBuilder& SetArribute(const std::string& name, const std::string& val);
  WorkerTable* WorkerTableBuild();
  ServerTable* ServerTableBuild();
private:
  std::unordered_map<std::string, std::string> params_;
};

}

#endif // MULTIVERSO_TABLE_INTERFACE_H_
