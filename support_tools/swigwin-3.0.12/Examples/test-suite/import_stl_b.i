%module import_stl_b

%import "import_stl_a.i"

%inline %{
#include <vector>
std::vector<int> process_vector(const std::vector<int>& v) {
  std::vector<int> v_new = v;
  v_new.push_back(4);
  return v_new;
}
%}

