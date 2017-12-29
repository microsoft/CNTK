/* File : example.h */

#include <map>
#include <string>

template<class Key, class Value>
std::map<Key,Value> half_map(const std::map<Key,Value>& v) {
  typedef typename std::map<Key,Value>::const_iterator iter;  
  std::map<Key,Value> w;
  for (iter i = v.begin(); i != v.end(); ++i) {
    w[i->first] = (i->second)/2;
  }  
  return w;
}



