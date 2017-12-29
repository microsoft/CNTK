/* File : example.cpp */

#include "example.h"

#include <iostream>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <sstream>


template <typename T>
std::list<T> concat_list(const std::list<T> list, const std::list<T> other_list)
{
    std::list<T> out_list(list);
    out_list.insert(out_list.end(), other_list.begin(), other_list.end());
    return out_list;
}

// int lists

std::list<int> create_integer_list(const int rangemin, const int rangemax)
{
    std::list<int> out_list;
    for (int i = rangemin; i <= rangemax; i++)
    {
        out_list.push_back(i);
    }
    return out_list;
}

int sum_integer_list(const std::list<int>& list)
{
    return std::accumulate(list.begin(), list.end(), 0);
}

std::list<int> concat_integer_list(const std::list<int> list, const std::list<int> other_list)
{
    return concat_list<int>(list, other_list);
}

// string lists

std::list<std::string> create_string_list(const char* svalue)
{
    std::list<std::string> out_list;
    std::string str(svalue);

    std::istringstream iss(str);
    std::copy(std::istream_iterator<std::string>(iss),
        std::istream_iterator<std::string>(),
        std::inserter<std::list<std::string> >(out_list, out_list.begin()));

    return out_list;
}

std::list<std::string> concat_string_list(const std::list<std::string> list, const std::list<std::string> other_list)
{
    return concat_list<std::string>(list, other_list);
}

