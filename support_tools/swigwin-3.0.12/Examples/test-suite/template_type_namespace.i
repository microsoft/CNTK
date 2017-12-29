%module template_type_namespace

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) std::vector<std::string>;	// Ruby, wrong class name

%include std_string.i
%include std_vector.i
 
%template(string_vector) std::vector<std::string>;
 
%inline %{
     std::vector<std::string> foo() {
         return std::vector<std::string>(1,"foo");
     }
%}                                                  
