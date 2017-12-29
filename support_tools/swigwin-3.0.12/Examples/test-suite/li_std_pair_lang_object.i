%module li_std_pair_lang_object

%include <std_pair.i>

namespace std {
  %template(ValuePair) pair< swig::LANGUAGE_OBJ, swig::LANGUAGE_OBJ >;
}

