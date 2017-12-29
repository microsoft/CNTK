// A speed test of the ruby stl
%module ruby_li_std_speed

%include <std_list.i>
%include <std_vector.i>
%include <std_deque.i>
%include <std_set.i>

%template(RbList)   std::list<swig::GC_VALUE>;
%template(RbVector) std::vector<swig::GC_VALUE>;
%template(RbDeque)  std::deque<swig::GC_VALUE>;
%template(RbSet)    std::set<swig::GC_VALUE>; 

%template(RbFloatList)   std::list<float>;
%template(RbFloatVector) std::vector<float>;
%template(RbFloatDeque)  std::deque<float>;
%template(RbFloatSet)    std::set<float>; 
