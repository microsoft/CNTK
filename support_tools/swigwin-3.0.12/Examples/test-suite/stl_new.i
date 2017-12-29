%module stl_new

%include <std_vector.i>
%include <std_deque.i>
%include <std_list.i>
%include <std_set.i>
%include <std_map.i>

%template(Vector  ) std::vector  <swig::LANGUAGE_OBJ>;
%template(Deque   ) std::deque   <swig::LANGUAGE_OBJ>;
%template(List    ) std::list    <swig::LANGUAGE_OBJ>;

%template(Set     ) std::set     <swig::LANGUAGE_OBJ,
				  swig::BinaryPredicate<> >;
%template(Map     ) std::map     <swig::LANGUAGE_OBJ,swig::LANGUAGE_OBJ,
                                   swig::BinaryPredicate<> >;


// %inline %{
//     namespace swig {
//         void nth_element(swig::Iterator_T< _Iter>& first,
//                          swig::Iterator_T< _Iter>& nth,
//                          swig::Iterator_T< _Iter>& last,
//                          const swig::BinaryPredicate<>& comp = swig::BinaryPredicate<>())
//         {
// 	  std::nth_element( first, nth, last, comp);
//         }
//     }
// %}
