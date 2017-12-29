%module li_std_functors

%include <std_vector.i>
%include <std_deque.i>
%include <std_list.i>
%include <std_set.i>
%include <std_map.i>
%include <std_functors.i>

%template(Vector  ) std::vector  <swig::LANGUAGE_OBJ>;
%template(Deque   ) std::deque   <swig::LANGUAGE_OBJ>;
%template(List    ) std::list    <swig::LANGUAGE_OBJ>;

%template(Set     ) std::set     <swig::LANGUAGE_OBJ,
                                   swig::BinaryPredicate<> >;
%template(Map     ) std::map     <swig::LANGUAGE_OBJ,swig::LANGUAGE_OBJ,
                                   swig::BinaryPredicate<> >;

