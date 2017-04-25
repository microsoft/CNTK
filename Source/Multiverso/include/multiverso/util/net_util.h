#ifndef MULTIVERSO_UTIL_NET_UTIL_H_
#define MULTIVERSO_UTIL_NET_UTIL_H_

#include <string>
#include <unordered_set>

namespace multiverso {
namespace net {

void GetLocalIPAddress(std::unordered_set<std::string>* result);

}  // namespace net
}  // namespace multiverso

#endif  // MULTIVERSO_UTIL_NET_UTIL_H_
