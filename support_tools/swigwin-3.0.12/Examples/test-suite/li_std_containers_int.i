%module li_std_containers_int

//
// Test containers of type int
//

%include std_vector.i
%include std_list.i

%template(vector_int) std::vector<int>;
%template(list_int) std::list<int>;

