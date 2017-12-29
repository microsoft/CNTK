%module ruby_keywords

// fix up conflicts with C++ keywords
%rename("and") Keywords::and_;
%rename("break") Keywords::break_;
%rename("case") Keywords::case_;
%rename("class") Keywords::class_;
%rename("defined?") Keywords::defined_;
%rename("do") Keywords::do_;
%rename("else") Keywords::else_;
%rename("false") Keywords::false_;
%rename("for") Keywords::for_;
%rename("if") Keywords::if_;
%rename("not") Keywords::not_;
%rename("return") Keywords::return_;
%rename("or") Keywords::or_;
%rename("true") Keywords::true_;
%rename("while") Keywords::while_;


%inline %{

class Keywords {
public:
  Keywords() {}

  const char* alias() { return "alias"; }
  const char* and_() { return "and"; }
  const char* begin() { return "begin"; }
  const char* break_() { return "break"; }
  const char* case_() { return "case"; }
  const char* class_() { return "class"; }
  const char* def() { return "def"; }
  const char* defined_() { return "defined?"; }
  const char* do_() { return "do"; }
  const char* else_() { return "else"; }
  const char* elsif() { return "elsif"; }
  const char* end() { return "end"; }
  const char* ensure() { return "ensure"; }
  const char* false_() { return "false"; }
  const char* for_() { return "for"; }
  const char* if_() { return "if"; }
  const char* in() { return "in"; }
  const char* module() { return "module"; }
  const char* next() { return "next"; }
  const char* nil() { return "nil"; }
  const char* not_() { return "not"; }
  const char* or_() { return "or"; }
  const char* redo() { return "redo"; }
  const char* rescue() { return "rescue"; }
  const char* retry() { return "retry"; }
  const char* return_() { return "return"; }
  const char* self() { return "self"; }
  const char* super() { return "super"; }
  const char* then() { return "then"; }
  const char* true_() { return "true"; }
  const char* undef() { return "undef"; }
  const char* under() { return "under"; }
  const char* unless() { return "unless"; }
  const char* until() { return "until"; }
  const char* when() { return "when"; }
  const char* while_() { return "while"; }
  const char* yield() { return "yield"; }
};
%}
