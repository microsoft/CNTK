#include "multiverso/dashboard.h"

#include <map>
#include <sstream>
#include <string>

#include "multiverso/util/log.h"

namespace multiverso {

std::map<std::string, Monitor*> Dashboard::record_;
std::mutex Dashboard::m_;

void Dashboard::AddMonitor(const std::string& name, Monitor* monitor) {
  std::lock_guard<std::mutex> l(m_);
  CHECK(record_[name] == nullptr);
  record_[name] = monitor;
}

void Dashboard::RemoveMonitor(const std::string& name) {
  std::lock_guard<std::mutex> l(m_);
  CHECK_NOTNULL(record_[name]);
  record_.erase(name);
}

std::string Dashboard::Watch(const std::string& name) {
  std::lock_guard<std::mutex> l(m_);
  std::string result;
  if (record_.find(name) == record_.end()) return result;
  Monitor* monitor = record_[name];
  CHECK_NOTNULL(monitor);
  return monitor->info_string();
}

std::string Monitor::info_string() const {
  std::ostringstream oss;
  oss << "[" << name_ << "] "
      << " count = " << count_
      << " elapse = " << elapse_ << "ms"
      << " average = " << average() << "ms";
  return oss.str();
}

void Dashboard::Display() {
  std::lock_guard<std::mutex> l(m_);
  Log::Info("--------------Show dashboard monitor information--------------\n");
  for (auto& it : record_) Log::Info("%s\n", it.second->info_string().c_str());
  Log::Info("--------------------------------------------------------------\n");
}

}  // namespace multiverso
