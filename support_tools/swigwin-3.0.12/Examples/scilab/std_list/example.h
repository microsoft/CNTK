/* File : example.h */

#include <list>
#include <string>


// integer lists
std::list<int> create_integer_list(const int size, const int value);
int sum_integer_list(const std::list<int>& list);
std::list<int> concat_integer_list(const std::list<int> list, const std::list<int> other_list);

// string lists
std::list<std::string> create_string_list(const char* value);
std::list<std::string> concat_string_list(const std::list<std::string> list, const std::list<std::string> other_list);
