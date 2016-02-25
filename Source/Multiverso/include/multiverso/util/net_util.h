#ifndef MULTIVERSO_UTIL_NET_UTIL_H_
#define MULTIVERSO_UTIL_NET_UTIL_H_

#include <string>
#include <unordered_set>

namespace multiverso {
namespace net {

//std::string GetHostName() {
//  return "";
//}
//
//std::string HostNameToIP(std::string hostname) {
//  return "";
//}
//
//std::string IPToHostName(std::string ip) {
//  return "";
//}
//
//bool IsLocalAddress(std::string ip) {
//  return true;
//}

void GetLocalIPAddress(std::unordered_set<std::string>* result);

} // namespace net
} // namespace multiverso

#endif // MULTIVERSO_UTIL_NET_UTIL_H_