/* This testcase checks the new wrappers for the new unordered_ STL types
   introduced in C++11. */
%module cpp11_hash_tables

%inline %{
#include <set>
//#include <map>
#include <unordered_set>
//#include <unordered_map>
%}

%include "std_set.i"
//%include "std_map.i"
%include "std_unordered_set.i"
//%include "std_unordered_map.i"
%template (SetInt) std::set<int>;
//%template (MapIntInt) std::map<int, int>;
%template (UnorderedSetInt) std::unordered_set<int>;
//%template (UnorderedMapIntInt) std::unordered_map<int, int>;

%inline %{
using namespace std;

class MyClass {
public:
  set<int> getSet() { return _set; }
  void addSet(int elt) { _set.insert(_set.begin(), elt); }
//  map<int, int> getMap() { return _map; }
//  void addMap(int elt1, int elt2) { _map.insert(make_pair(elt1, elt2)); }

  unordered_set<int> getUnorderedSet() { return _unordered_set; }
  void addUnorderedSet(int elt) { _unordered_set.insert(_unordered_set.begin(), elt); }
//  unordered_map<int, int> getUnorderedMap() { return _unordered_map; }
//  void addUnorderedMap(int elt1, int elt2) { _unordered_map.insert(make_pair(elt1, elt2)); }
private:
  set<int> _set;
//  map<int, int> _map;

  unordered_set<int> _unordered_set;
//  unordered_map<int, int> _unordered_map;
};
%}

