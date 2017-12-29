%module(extranative="1") python_extranative

%include <std_vector.i>
%include <std_string.i>

%template(VectorString) std::vector<std::string>;

%inline %{
std::vector<std::string> make_vector_string() {
  std::vector<std::string> vs;
  vs.push_back("one");
  vs.push_back("two");
  return vs;
}
%}

