%module(threads=1) python_threads

%include <std_vector.i>

%inline %{
struct Action {
  int val;
  Action(int val = 0) : val(val) {}
};
%}

%template(VectorActionPtr) std::vector<Action *>;

%inline %{
#include <vector>
#include <iostream>
template <typename T> struct myStlVector : public std::vector<T> {
};
typedef myStlVector <Action *> ActionList;

%}

%template(ActionList) myStlVector<Action *>;

%inline %{
class ActionGroup
{
public:
  ActionList &GetActionList () const {
    static ActionList list;
    list.push_back(new Action(1));
    list.push_back(new Action(2));
    list.push_back(new Action(3));
    list.push_back(new Action(4));
    return list;
  }
};
%}


